from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def run_migrations():
    """Add new columns to existing tables if they don't exist (SQLite safe)."""
    new_columns = [
        ("training_experiments", "process_pid", "INTEGER"),
        ("training_experiments", "log_file", "VARCHAR(500)"),
        ("training_experiments", "error_message", "TEXT"),
        ("training_experiments", "total_steps", "INTEGER"),
        ("training_experiments", "current_step", "INTEGER"),
        ("extraction_jobs", "base_url", "VARCHAR(500)"),
        ("extraction_jobs", "api_key", "VARCHAR(500)"),
        ("extraction_jobs", "task_type", "VARCHAR(50) DEFAULT 'qa_extraction'"),
        ("extraction_jobs", "assistant_id", "INTEGER"),
        ("llm_assistants", "gpu_ids", "JSON"),
        ("llm_assistants", "extra_env_vars", "JSON"),
        ("evaluation_runs", "phase", "VARCHAR(30) DEFAULT 'pending'"),
    ]
    with engine.connect() as conn:
        for table, col, col_type in new_columns:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                pass  # Column already exists

    # On startup, any local assistant marked "running" / "starting" is stale
    # because we lost the subprocess handle when the backend restarted.
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE llm_assistants SET status='stopped', process_pid=NULL, port=NULL "
                "WHERE type='local' AND status IN ('running', 'starting')"
            ))
            conn.commit()
    except Exception:
        pass
