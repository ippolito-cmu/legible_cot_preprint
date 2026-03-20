#!/usr/bin/env python3
"""
Unified Database Migration Script
==================================

This script consolidates ALL migration functionality into one place:
- Migrates traces, efficiency, and redundancy metrics
- Migrates pedagogical utility (PU) data 
- Builds cache tables for fast PU visualization (percentage + absolute bins)
- Supports incremental updates (only migrates new data)
- Can be run on specific datasets/models or all data

Usage:
    # Migrate everything (recommended for initial setup)
    python migrate_database.py --all
    
    # Migrate specific dataset
    python migrate_database.py --dataset math
    
    # Migrate only PU data
    python migrate_database.py --pu-only
    
    # Rebuild caches only (after new PU data)
    python migrate_database.py --rebuild-cache
    
    # Force full re-migration (deletes existing data first)
    python migrate_database.py --all --force
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from viz.db_backend import migrate_pickle_to_db, migrate_pu_data, migrate_backtracking_data, TraceDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(title):
    """Print a nice banner"""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()
    sys.stdout.flush()  # Force immediate output in SLURM logs


def get_db_stats(db: TraceDB):
    """Get database statistics"""
    cursor = db.conn.cursor()
    
    stats = {}
    
    # Traces
    cursor.execute("SELECT COUNT(*) as count FROM traces")
    stats['traces'] = cursor.fetchone()['count']
    
    # PU tables - just count tables, not records (too slow)
    cursor.execute("""
        SELECT COUNT(*) as count FROM sqlite_master 
        WHERE type='table' AND name LIKE 'pu_%'
        AND name NOT LIKE 'pu_%_bins'
    """)
    stats['pu_tables'] = cursor.fetchone()['count']
    stats['pu_records'] = 0  # Skip slow count
    stats['pu_students'] = stats['pu_tables']  # Each table = one student
    stats['pu_teachers'] = 0  # Skip slow count
    
    # Cache tables
    try:
        cursor.execute("SELECT COUNT(*) as count FROM pu_percentage_bins")
        stats['pct_bins'] = cursor.fetchone()['count']
    except:
        stats['pct_bins'] = 0
    
    try:
        cursor.execute("SELECT COUNT(*) as count FROM pu_absolute_bins")
        stats['abs_bins'] = cursor.fetchone()['count']
    except:
        stats['abs_bins'] = 0
    
    return stats


def print_stats(db: TraceDB):
    """Print database statistics"""
    stats = get_db_stats(db)
    
    print("\n📊 Database Statistics:")
    print(f"   Traces:           {stats['traces']:,}")
    print(f"   PU Tables:        {stats['pu_tables']}")
    print(f"   PU Students:      {stats['pu_students']}")
    print(f"   Percentage Bins:  {stats['pct_bins']:,}")
    print(f"   Absolute Bins:    {stats['abs_bins']:,}")
    print()


def migrate_teacher_scores(db: TraceDB, dataset_name=None):
    """
    Populate teacher_score column in pedagogical_utility table.
    This is a one-time migration to denormalize teacher scores for fast cache building.
    
    Args:
        db: Database connection
        dataset_name: Specific dataset to migrate, or None for all
    """
    print_banner("Migrating Teacher Scores (Denormalization)")
    
    cursor = db.conn.cursor()
    
    # Check if column exists
    cursor.execute("PRAGMA table_info(pedagogical_utility)")
    columns = [row['name'] for row in cursor.fetchall()]
    
    if 'teacher_score' not in columns:
        print("Adding teacher_score column...")
        cursor.execute("ALTER TABLE pedagogical_utility ADD COLUMN teacher_score REAL")
        db.conn.commit()
        print("✓ Column added")
    
    # Check how many records need migration
    cursor.execute("SELECT COUNT(*) as count FROM pedagogical_utility WHERE teacher_score IS NULL")
    null_count = cursor.fetchone()['count']
    
    if null_count == 0:
        print("✓ All teacher scores already populated")
        return
    
    print(f"Found {null_count:,} records to populate")
    sys.stdout.flush()
    
    # Populate teacher_score using temp table with JOIN (much faster than correlated subquery)
    print(f"Creating temporary lookup table...")
    sys.stdout.flush()
    
    # Create temp table with teacher scores
    cursor.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS temp_teacher_scores AS
        SELECT 
            pu.id as pu_id,
            t.score as teacher_score
        FROM pedagogical_utility pu
        INNER JOIN traces t ON 
            t.dataset = pu.dataset 
            AND t.model = pu.teacher_model 
            AND t.trace_index = pu.trace_index
        WHERE pu.teacher_score IS NULL
    """)
    
    temp_count = cursor.execute("SELECT COUNT(*) FROM temp_teacher_scores").fetchone()[0]
    print(f"  Created lookup for {temp_count:,} records")
    sys.stdout.flush()
    
    # Now do fast update from temp table
    print(f"Updating pedagogical_utility table...")
    sys.stdout.flush()
    
    cursor.execute("""
        UPDATE pedagogical_utility
        SET teacher_score = (
            SELECT teacher_score 
            FROM temp_teacher_scores 
            WHERE temp_teacher_scores.pu_id = pedagogical_utility.id
        )
        WHERE id IN (SELECT pu_id FROM temp_teacher_scores)
    """)
    
    db.conn.commit()
    
    # Drop temp table
    cursor.execute("DROP TABLE temp_teacher_scores")
    
    print(f"✓ Populated {cursor.rowcount:,} teacher scores")
    sys.stdout.flush()
    
    # Create index if it doesn't exist
    print("Creating index on teacher_score...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pu_teacher_score 
        ON pedagogical_utility(dataset, teacher_model, student_model, teacher_score)
    """)
    db.conn.commit()
    print("✓ Index created")


def rebuild_cache(db: TraceDB, dataset_name=None):
    """Rebuild percentage and absolute bin caches from split PU tables
    
    Args:
        db: Database connection
        dataset_name: Specific dataset to rebuild, or None for all
    """
    import numpy as np
    from collections import defaultdict
    
    print_banner("Rebuilding Cache Tables (Split Table Strategy)")
    
    cursor = db.conn.cursor()
    
    # Get all PU tables
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'pu_%'
        AND name NOT LIKE 'pu_%_bins'
    """)
    pu_tables = [row['name'] for row in cursor.fetchall()]
    
    logger.info(f"Found {len(pu_tables)} PU tables")
    print(f"📦 Processing {len(pu_tables)} PU tables...")
    sys.stdout.flush()
    
    # Hardcode known teachers to avoid slow DISTINCT queries on 2M row tables
    # These names match the actual format in the PU tables (underscores, not slashes)
    KNOWN_TEACHERS = [
        'Qwen_QwQ-32B',
        'Qwen_Qwen3-0.6B',
        'Qwen_Qwen3-32B',
        'Qwen_Qwen3-4B',
        'Qwen_Qwen3-8B',
        'deepseek-ai_DeepSeek-R1-Distill-Qwen-32B',
        'deepseek-ai_deepseek-r1-0528',
        'deepseek-r1-0528',
        'google_gemma-3-12b-it',
        'google_gemma-3-27b-it',
        'gpt-5',
        'meta-llama_Meta-Llama-3.1-70B-Instruct',
        'meta-llama_Meta-Llama-3.1-8B-Instruct',
        'mistralai_Magistral-Small-2509',
        'nvidia_Llama-3.1-Nemotron-Nano-8B-v1',
        'nvidia_OpenReasoning-Nemotron-32B',
        'openai_gpt-oss-120b',
        'openai_gpt-oss-20b'
    ]
    
    # For each PU table, use known teachers list
    all_combinations = []
    for i, table_name in enumerate(pu_tables, 1):
        # Extract dataset and student from table name: pu_math_llama_3_2_1b_instruct
        parts = table_name.split('_', 2)  # Split into ['pu', 'dataset', 'student']
        if len(parts) < 3:
            continue
        dataset = parts[1]
        student_short = parts[2]
        
        # Skip if not the requested dataset
        if dataset_name and dataset != dataset_name:
            continue
        
        print(f"[{i}/{len(pu_tables)}] Processing {table_name}...")
        sys.stdout.flush()
        
        # Use hardcoded teacher list instead of slow DISTINCT query
        teachers = KNOWN_TEACHERS
        print(f"  → Using {len(teachers)} known teachers")
        sys.stdout.flush()
        
        for teacher_model in teachers:
            all_combinations.append((dataset, teacher_model, table_name, student_short))
    
    logger.info(f"Found {len(all_combinations)} dataset/teacher combinations to cache")
    print(f"\n✓ Total: {len(all_combinations)} combinations to cache")
    sys.stdout.flush()
    
    def get_short_name(full_name):
        """Extract short name from full model path"""
        if '/' in full_name:
            parts = full_name.split('/')[-1]
        else:
            parts = full_name
        
        for suffix in ['-Instruct', '-instruct', '-Chat', '-chat']:
            if parts.endswith(suffix):
                parts = parts[:-len(suffix)]
        
        return parts
    
    # Process each combination
    for i, (dataset, teacher_model, table_name, student_short) in enumerate(all_combinations, 1):
        teacher_short_name = get_short_name(teacher_model)
        
        progress_pct = (i / len(all_combinations)) * 100
        print(f"\n[{i}/{len(all_combinations)} - {progress_pct:.1f}%] Caching {dataset}/{teacher_model}/{student_short}")
        logger.info(f"[{i}/{len(all_combinations)}] Caching {dataset}/{teacher_model}/{student_short}")
        sys.stdout.flush()
        
        try:
            print(f"  → Building percentage bins...")
            sys.stdout.flush()
            # Build percentage bins cache from split table
            _build_percentage_cache_split(db, dataset, teacher_model, teacher_short_name, student_short, table_name)
            
            print(f"  → Building absolute bins...")
            sys.stdout.flush()
            # Build absolute bins cache from split table
            _build_absolute_cache_split(db, dataset, teacher_model, teacher_short_name, student_short, table_name)

            print(f"  → Building regression cache...")
            sys.stdout.flush()
            _build_pu_regression_cache_split(db, dataset, teacher_model, teacher_short_name, student_short, table_name)
            
            print(f"  ✓ Complete")
            sys.stdout.flush()
            
        except Exception as e:
            import traceback
            logger.error(f"Error caching {dataset}/{teacher_model}/{student_short}: {e}")
            logger.error(traceback.format_exc())
            print(f"  ✗ Error: {e}")
            print(traceback.format_exc())
            sys.stdout.flush()
            continue
    
    print("\n✓ Cache rebuild complete!")
    logger.info("Cache rebuild complete!")
    sys.stdout.flush()


def _build_pu_regression_cache_split(db, dataset, teacher_model, teacher_short_name, student_short, table_name):
    """Build regression cache from split PU table.

    Regression definition matches src/scripts/analyze_metric_correlations.py:
    a regression is a transition where previous step is correct and next step is incorrect.
    """
    cursor = db.conn.cursor()

    cursor.execute(
        f"""
        WITH step_means AS (
            SELECT
                trace_index,
                num_steps,
                AVG(CASE WHEN score > 0 THEN 1.0 ELSE 0.0 END) AS mean_correct
            FROM {table_name}
            WHERE teacher_model = ? AND teacher_score > 0.5
            GROUP BY trace_index, num_steps
        ),
        ordered AS (
            SELECT
                trace_index,
                num_steps,
                mean_correct,
                LAG(mean_correct) OVER (PARTITION BY trace_index ORDER BY num_steps) AS prev_correct
            FROM step_means
        ),
        events AS (
            SELECT
                trace_index,
                SUM(CASE WHEN prev_correct >= 0.5 AND mean_correct < 0.5 THEN 1 ELSE 0 END) AS regressions
            FROM ordered
            GROUP BY trace_index
        )
        SELECT
            COUNT(*) AS total_questions,
            SUM(CASE WHEN regressions > 0 THEN 1 ELSE 0 END) AS regressed_questions,
            SUM(regressions) AS total_regressions
        FROM events
        """,
        (teacher_model,),
    )

    row = cursor.fetchone()
    if not row:
        return

    total_questions = int(row['total_questions'] or 0)
    regressed_questions = int(row['regressed_questions'] or 0)
    total_regressions = int(row['total_regressions'] or 0)

    regression_percentage = (100.0 * regressed_questions / total_questions) if total_questions else 0.0
    avg_regressions_per_question = (total_regressions / total_questions) if total_questions else 0.0

    cursor.execute(
        """
        INSERT OR REPLACE INTO pu_regression_cache
        (dataset, teacher_model, teacher_short_name, student_model,
         total_questions, regressed_questions, regression_percentage,
         total_regressions, avg_regressions_per_question)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset,
            teacher_model,
            teacher_short_name,
            student_short,
            total_questions,
            regressed_questions,
            float(regression_percentage),
            total_regressions,
            float(avg_regressions_per_question),
        ),
    )
    db.conn.commit()


def _build_percentage_cache_split(db, dataset, teacher_model, teacher_short_name, student_short, table_name):
    """Build percentage bin cache from split PU table"""
    import numpy as np
    
    cursor = db.conn.cursor()
    
    # Normalize connections scores
    score_divisor = 4.0 if dataset.lower() == 'connections' else 1.0
    
    # Query split table directly - MUCH faster than old JOIN approach!
    # Filter by teacher_score > 0.5 (only include traces where teacher was correct)
    cursor.execute(f"""
        SELECT 
            CAST(percentage_complete / 2.0 AS INT) * 2.0 + 1.0 as bin_center,
            score
        FROM {table_name}
        WHERE teacher_model = ? AND teacher_score > 0.5
        ORDER BY bin_center
    """, (teacher_model,))
    
    rows = cursor.fetchall()
    
    # Group scores by bin and normalize
    bin_scores = {}
    for row in rows:
        bin_center = row['bin_center']
        score = row['score'] / score_divisor
        if bin_center not in bin_scores:
            bin_scores[bin_center] = []
        bin_scores[bin_center].append(score)
    
    # Compute statistics and insert
    for bin_center, scores in bin_scores.items():
        mean_correct = np.mean(scores)
        std_correct = np.std(scores) if len(scores) > 1 else 0.0
        count = len(scores)
        
        cursor.execute("""
            INSERT OR REPLACE INTO pu_percentage_bins
            (dataset, teacher_model, teacher_short_name, student_model, bin_center, mean_correct, std_correct, count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (dataset, teacher_model, teacher_short_name, student_short, bin_center, mean_correct, std_correct, count))
    
    db.conn.commit()


def _build_absolute_cache_split(db, dataset, teacher_model, teacher_short_name, student_short, table_name):
    """Build absolute bin cache from split PU table"""
    import numpy as np
    from collections import defaultdict
    
    cursor = db.conn.cursor()
    
    # Normalize connections scores
    score_divisor = 4.0 if dataset.lower() == 'connections' else 1.0
    
    # Query split table directly - MUCH faster!
    # Filter by teacher_score > 0.5 (only include traces where teacher was correct)
    cursor.execute(f"""
        SELECT num_steps, score
        FROM {table_name}
        WHERE teacher_model = ? AND teacher_score > 0.5
        ORDER BY num_steps
    """, (teacher_model,))
    
    rows = cursor.fetchall()
    
    if not rows:
        return
    
    # Use bin_size = 10
    bin_size = 10
    max_steps = max(row['num_steps'] for row in rows)
    
    # Group scores into bins
    bins = defaultdict(list)
    for row in rows:
        num_steps = row['num_steps']
        score = row['score'] / score_divisor
        bin_idx = int(num_steps // bin_size)
        bin_start = bin_idx * bin_size
        bin_center = bin_start + (bin_size / 2.0)
        bins[bin_center].append(score)
    
    # Filter bins with >= 10 samples
    filtered_bins = {k: v for k, v in bins.items() if len(v) >= 10}
    
    # Insert aggregated data
    for bin_center, scores in filtered_bins.items():
        mean_correct = np.mean(scores)
        std_correct = np.std(scores) if len(scores) > 1 else 0.0
        count = len(scores)
        
        cursor.execute("""
            INSERT OR REPLACE INTO pu_absolute_bins
            (dataset, teacher_model, teacher_short_name, student_model, bin_center, mean_correct, std_correct, count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (dataset, teacher_model, teacher_short_name, student_short, bin_center, mean_correct, std_correct, count))
    
    db.conn.commit()


def _old_build_percentage_cache(db, dataset, teacher_model, teacher_short_name, student_model):
    """OLD: Build percentage bins cache (includes student dimension)"""
    # This function is now replaced by _build_percentage_cache_split
    # Kept for reference only
    pass


def _old_build_absolute_cache_with_student(db, dataset, teacher_model, teacher_short_name, student_model):
    """OLD: Build absolute bin cache for specific teacher/student combination"""
    # This function is now replaced by _build_absolute_cache_split
    # Kept for reference only
    pass


def _old_migrate_teacher_scores(db: TraceDB, dataset_name=None):
    """OLD: Populate teacher_score column - no longer needed with split tables"""
    # Teacher scores are now populated inline during PU migration
    # Kept for reference only
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Unified database migration script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # What to migrate
    parser.add_argument('--all', action='store_true',
                       help='Migrate all data (traces + PU + rebuild cache)')
    parser.add_argument('--traces-only', action='store_true',
                       help='Migrate only traces/efficiency/redundancy')
    parser.add_argument('--pu-only', action='store_true',
                       help='Migrate only pedagogical utility data')
    parser.add_argument('--backtracking-only', action='store_true',
                       help='Migrate only backtracking analysis data')
    parser.add_argument('--rebuild-cache', action='store_true',
                       help='Only rebuild cache tables (skip migration)')
    
    # Filters
    parser.add_argument('--dataset', type=str,
                       help='Migrate specific dataset only (connections/gpqa/math)')
    parser.add_argument('--model', type=str,
                       help='Migrate specific model only (requires --dataset)')
    
    # Options
    parser.add_argument('--force', action='store_true',
                       help='Force re-migration (delete existing database)')
    parser.add_argument('--skip-cache', action='store_true',
                       help='Skip cache rebuild after migration')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model and not args.dataset:
        parser.error('--model requires --dataset')
    
    if not any([args.all, args.traces_only, args.pu_only, args.backtracking_only, args.rebuild_cache]):
        parser.error('Must specify one of: --all, --traces-only, --pu-only, --backtracking-only, --rebuild-cache')
    
    # Start
    start_time = datetime.now()
    print_banner("Unified Database Migration")
    
    db_path = PROJECT_ROOT / 'outputs' / 'traces.db'
    
    # Handle force flag
    if args.force and db_path.exists():
        logger.warning(f"Deleting existing database: {db_path}")
        backup_path = db_path.parent / f"traces.db.backup_{start_time.strftime('%Y%m%d_%H%M%S')}"
        db_path.rename(backup_path)
        logger.info(f"Backup created: {backup_path}")
    
    # Initialize database
    db = TraceDB(str(db_path))
    db.initialize_schema()
    
    # Get initial stats
    if db_path.exists():
        logger.info("Database exists. Will perform incremental update.")
        print_stats(db)
    
    # Migrate traces
    if args.all or args.traces_only:
        print_banner("Migrating Traces + Efficiency + Redundancy")
        try:
            # Efficiency computed inline; redundancy loaded from pickle files
            migrate_pickle_to_db(
                dataset_name=args.dataset, 
                model_name=args.model
            )
            logger.info("✓ Trace migration complete")
        except Exception as e:
            logger.error(f"✗ Trace migration failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Migrate PU data
    if args.all or args.pu_only:
        print_banner("Migrating Pedagogical Utility Data")
        try:
            migrate_pu_data(dataset_name=args.dataset)
            logger.info("✓ PU migration complete")
        except Exception as e:
            logger.error(f"✗ PU migration failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Migrate backtracking analysis data
    if args.all or args.backtracking_only:
        print_banner("Migrating Backtracking Analysis Data")
        try:
            migrate_backtracking_data(dataset_name=args.dataset, model_name=args.model)
            logger.info("✓ Backtracking migration complete")
        except Exception as e:
            logger.error(f"✗ Backtracking migration failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Note: teacher_score is now populated inline during PU migration (no separate step needed)
    
    # Rebuild cache
    if (args.all or args.pu_only or args.rebuild_cache) and not args.skip_cache:
        try:
            rebuild_cache(db, dataset_name=args.dataset)
            logger.info("✓ Cache rebuild complete")
        except Exception as e:
            logger.error(f"✗ Cache rebuild failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Final stats
    print_banner("Migration Complete!")
    print_stats(db)
    
    db_size_mb = db_path.stat().st_size / 1024 / 1024
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"📁 Database: {db_path}")
    print(f"💾 Size: {db_size_mb:.1f} MB")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print()
    print("🚀 Next steps:")
    print("   1. Start Flask server: python src/viz/app_db.py")
    print("   2. Visit: http://localhost:8080")
    print()
    
    db.close()


if __name__ == '__main__':
    main()
