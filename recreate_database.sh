
set -e

echo "========================================="
echo "Database Recreation Script"
echo "========================================="
echo

DB_PATH="outputs/traces.db"
BACKUP_PATH="outputs/traces.db.backup_$(date +%Y%m%d_%H%M%S)"

if [ -f "$DB_PATH" ]; then
    echo "Found existing database: $DB_PATH"

    echo "Creating backup at $BACKUP_PATH..."
    cp "$DB_PATH" "$BACKUP_PATH"
    echo "✓ Backup created"
    echo "Deleting old database..."
    rm "$DB_PATH"
    echo "✓ Database deleted"
else
    echo "No existing database found at $DB_PATH"
fi

echo
echo "========================================="
echo "Starting fresh migration"
echo "========================================="
echo

echo "Stopping any running Flask servers..."
pkill -f "python.*app_db.py" || true
pkill -f "python.*app.py" || true
sleep 2

echo "Running unified migration script..."
python migrate_database.py --all
echo "Done"
