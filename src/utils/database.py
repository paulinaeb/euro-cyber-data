"""Database utilities"""

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_batch
from contextlib import contextmanager
import logging

from src.utils.config import DB_CONFIG

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database manager"""
    
    def __init__(self):
        self.config = DB_CONFIG
    
    @contextmanager
    def get_connection(self):
        """Context manager for psycopg2 connection"""
        conn = psycopg2.connect(**self.config)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    @contextmanager
    def get_cursor(self, dict_cursor=True):
        """Context manager for database cursor"""
        with self.get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params=None, fetch=True):
        """Execute a query and optionally fetch results"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return None
    
    def execute_many(self, query: str, data_list: list, batch_size: int = 1000):
        """Execute batch insert/update"""
        with self.get_cursor() as cursor:
            execute_batch(cursor, query, data_list, page_size=batch_size)
        logger.info(f"Executed batch of {len(data_list)} operations")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                logger.info("✓ Database connection successful")
                return result[0] == 1
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            return False


# Global database instance
db = Database()


def ensure_database_exists():
    """Ensure the configured database exists, creating it if needed."""
    target_db = DB_CONFIG["database"]
    admin_config = DB_CONFIG.copy()
    admin_config["database"] = "postgres"

    conn = psycopg2.connect(**admin_config)
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            if cursor.fetchone():
                return False
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
            return True
    finally:
        conn.close()
