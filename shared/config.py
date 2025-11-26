"""
shared/config.py
Shared configuration accessible to BOTH validator and arc_saga.
This is the single source of truth for paths and settings.

Located at repository root so both modules can import easily.
Follows: Single Responsibility Principle, DRY principle
"""

from pathlib import Path
import os
from typing import Optional


class SharedConfig:
    """
    Centralized configuration for both validator and arc_saga modules.
    
    Both modules import from this single location - ensures consistency
    and prevents duplicate configuration.
    """
    
    # ========================================================================
    # PROJECT PATHS
    # ========================================================================
    
    # Repository root (where pyproject.toml is)
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # Module roots
    VALIDATOR_ROOT: Path = PROJECT_ROOT / "ai_config_validator"
    ARC_SAGA_ROOT: Path = PROJECT_ROOT / "arc_saga"
    SHARED_ROOT: Path = PROJECT_ROOT / "shared"
    
    # ========================================================================
    # STORAGE CONFIGURATION
    # ========================================================================
    
    # User home directory storage (Windows-friendly)
    STORAGE_DIR: Path = Path(
        os.getenv("ARC_SAGA_STORAGE", "~\\.arc-saga")
    ).expanduser()
    
    DB_PATH: Path = STORAGE_DIR / "memory.db"
    FILES_DIR: Path = STORAGE_DIR / "files"
    LOGS_DIR: Path = STORAGE_DIR / "logs"
    
    # ========================================================================
    # LOGGING CONFIGURATION
    # ========================================================================
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = LOGS_DIR / "arc_saga.log"
    STRUCTURED_LOGGING: bool = True
    
    # ========================================================================
    # MONITORING PATHS (Log files to watch)
    # ========================================================================
    
    # Validator logs (arc_saga will monitor these)
    VALIDATOR_LOG_PATH: Optional[Path] = (
        VALIDATOR_ROOT / "logs"
        if VALIDATOR_ROOT.exists()
        else None
    )
    
    # Antigravity logs (Windows path)
    ANTIGRAVITY_LOG_PATH: Path = Path(
        os.getenv(
            "ANTIGRAVITY_LOG_PATH",
            "~\\AppData\\Roaming\\Antigravity\\logs"
        )
    ).expanduser()
    
    # ========================================================================
    # DATABASE CONFIGURATION
    # ========================================================================
    
    DB_TIMEOUT: int = 30  # SQLite connection timeout (seconds)
    DB_CHECK_SAME_THREAD: bool = False  # Allow multi-threaded access
    
    # ========================================================================
    # AUTO-TAGGING CONFIGURATION
    # ========================================================================
    
    MAX_TAGS_PER_MESSAGE: int = 5
    MAX_TAGS_PER_FILE: int = 7
    
    # ========================================================================
    # METHODS
    # ========================================================================
    
    @classmethod
    def initialize_dirs(cls) -> None:
        """
        Create all necessary directories if they don't exist.
        
        Safe to call multiple times.
        Raises PermissionError if unable to create directories.
        """
        try:
            cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            cls.FILES_DIR.mkdir(parents=True, exist_ok=True)
            cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create storage directories. Check permissions on {cls.STORAGE_DIR}: {e}"
            )
    
    @classmethod
    def get_validator_log_path(cls) -> Optional[Path]:
        """
        Get validator log path if it exists.
        
        Returns:
            Path object if validator logs exist, None otherwise
        """
        if cls.VALIDATOR_LOG_PATH and cls.VALIDATOR_LOG_PATH.exists():
            return cls.VALIDATOR_LOG_PATH
        return None
    
    @classmethod
    def get_antigravity_log_path(cls) -> Path:
        """
        Get Antigravity log path.
        
        Returns:
            Path object (may or may not exist)
        """
        return cls.ANTIGRAVITY_LOG_PATH
    
    @classmethod
    def validate_config(cls) -> list[str]:
        """
        Validate that all required directories are accessible.
        
        Returns:
            List of error messages (empty list if all valid)
            
        Example:
            >>> errors = SharedConfig.validate_config()
            >>>  if errors:
            ...     print("\\n".join(errors))
        """
        errors: list[str] = []
        
        # Check storage directory can be created
        try:
            cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            errors.append(f"Cannot create storage directory: {cls.STORAGE_DIR}")
        except Exception as e:
            errors.append(f"Storage directory error: {e}")
        
        # Check validator root is valid if it exists
        if cls.VALIDATOR_ROOT.exists() and not cls.VALIDATOR_ROOT.is_dir():
            errors.append(f"Validator root is not a directory: {cls.VALIDATOR_ROOT}")
        
        # Check arc_saga root is valid
        if cls.ARC_SAGA_ROOT.exists() and not cls.ARC_SAGA_ROOT.is_dir():
            errors.append(f"Arc Saga root is not a directory: {cls.ARC_SAGA_ROOT}")
        
        return errors
    
    @classmethod
    def get_python_path_additions(cls) -> list[Path]:
        """
        Get paths to add to sys.path for imports to work.
        
        Returns:
            List of Path objects to add to sys.path
            
        Note:
            In a proper monorepo, sys.path should already include PROJECT_ROOT,
            so this is mainly for edge cases.
        """
        return [cls.PROJECT_ROOT]
    
    @classmethod
    def __repr__(cls) -> str:
        """String representation for debugging."""
        return f"SharedConfig(root={cls.PROJECT_ROOT}, storage={cls.STORAGE_DIR})"


# ============================================================================
# MODULE-LEVEL INITIALIZATION
# ============================================================================

# Ensure directories exist when module is imported
SharedConfig.initialize_dirs()
