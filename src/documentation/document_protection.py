"""
Document Protection System for Feature Documentation Enhancement

This module provides comprehensive protection mechanisms for critical documentation files,
including backup creation, version control, change validation, and rollback capabilities.
"""

import os
import shutil
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import difflib


@dataclass
class BackupResult:
    """Result of a backup operation"""
    success: bool
    backup_path: str
    timestamp: str
    original_hash: str
    error_message: Optional[str] = None


@dataclass
class ChangeValidation:
    """Result of change validation"""
    is_valid: bool
    is_enhancement: bool
    preserved_content: bool
    added_content: List[str]
    removed_content: List[str]
    warnings: List[str]
    error_message: Optional[str] = None


@dataclass
class RollbackPoint:
    """Represents a rollback point"""
    id: str
    timestamp: str
    file_path: str
    backup_path: str
    description: str
    content_hash: str


class DocumentProtectionError(Exception):
    """Base exception for document protection errors"""
    pass


class BackupCreationError(DocumentProtectionError):
    """Raised when backup creation fails"""
    pass


class ChangeValidationError(DocumentProtectionError):
    """Raised when changes would cause data loss"""
    pass


class RollbackError(DocumentProtectionError):
    """Raised when rollback fails"""
    pass


class DocumentProtectionSystem:
    """
    Comprehensive document protection system that provides:
    - Timestamped backups in docs/research/case_study/
    - Change validation to prevent accidental overwrites
    - Version control and rollback capabilities
    - Content preservation verification
    """
    
    def __init__(self, backup_base_dir: str = "docs/research/case_study"):
        """
        Initialize the document protection system
        
        Args:
            backup_base_dir: Base directory for storing backups
        """
        self.backup_base_dir = Path(backup_base_dir)
        self.backup_dir = self.backup_base_dir / "backups"
        self.version_dir = self.backup_base_dir / "versions"
        self.rollback_dir = self.backup_base_dir / "rollback_points"
        
        # Create directory structure
        self._ensure_directory_structure()
        
        # Load or create metadata
        self.metadata_file = self.backup_base_dir / "protection_metadata.json"
        self.metadata = self._load_metadata()
        
        # Save metadata to ensure file exists
        self._save_metadata()
    
    def _ensure_directory_structure(self) -> None:
        """Create the required directory structure for backups and version control"""
        directories = [
            self.backup_base_dir,
            self.backup_dir,
            self.version_dir,
            self.rollback_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load protection metadata or create new if doesn't exist"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load metadata, creating new: {e}")
        
        return {
            "created_at": datetime.now().isoformat(),
            "backups": {},
            "versions": {},
            "rollback_points": {},
            "protected_files": []
        }
    
    def _save_metadata(self) -> None:
        """Save protection metadata to disk"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise DocumentProtectionError(f"Failed to save metadata: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except IOError as e:
            raise DocumentProtectionError(f"Failed to calculate hash for {file_path}: {e}")
    
    def _generate_timestamp(self) -> str:
        """Generate timestamp string for backup naming"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_backup(self, doc_path: str, description: str = "") -> BackupResult:
        """
        Create timestamped backup of document in docs/research/case_study/backups/
        
        Args:
            doc_path: Path to document to backup
            description: Optional description of the backup
            
        Returns:
            BackupResult with backup details
        """
        try:
            doc_path = Path(doc_path)
            
            if not doc_path.exists():
                return BackupResult(
                    success=False,
                    backup_path="",
                    timestamp="",
                    original_hash="",
                    error_message=f"Source file does not exist: {doc_path}"
                )
            
            # Generate backup filename with timestamp and microseconds for uniqueness
            import time
            timestamp = self._generate_timestamp()
            microseconds = str(int(time.time() * 1000000))[-6:]
            backup_filename = f"{doc_path.stem}_{timestamp}_{microseconds}{doc_path.suffix}"
            backup_path = self.backup_dir / backup_filename
            
            # Calculate original file hash
            original_hash = self._calculate_file_hash(str(doc_path))
            
            # Create backup
            shutil.copy2(str(doc_path), str(backup_path))
            
            # Verify backup integrity
            backup_hash = self._calculate_file_hash(str(backup_path))
            if original_hash != backup_hash:
                raise BackupCreationError("Backup integrity check failed")
            
            # Update metadata
            backup_info = {
                "original_path": str(doc_path),
                "backup_path": str(backup_path),
                "timestamp": timestamp,
                "description": description,
                "original_hash": original_hash,
                "backup_hash": backup_hash,
                "created_at": datetime.now().isoformat()
            }
            
            if str(doc_path) not in self.metadata["backups"]:
                self.metadata["backups"][str(doc_path)] = []
            
            self.metadata["backups"][str(doc_path)].append(backup_info)
            self._save_metadata()
            
            return BackupResult(
                success=True,
                backup_path=str(backup_path),
                timestamp=timestamp,
                original_hash=original_hash
            )
            
        except Exception as e:
            return BackupResult(
                success=False,
                backup_path="",
                timestamp="",
                original_hash="",
                error_message=f"Backup creation failed: {str(e)}"
            )
    
    def validate_changes(self, old_content: str, new_content: str, 
                        file_path: str = "") -> ChangeValidation:
        """
        Validate that changes are enhancements, not replacements
        
        Args:
            old_content: Original file content
            new_content: New file content
            file_path: Path to file being changed (for context)
            
        Returns:
            ChangeValidation result
        """
        try:
            # Basic validation
            if not new_content.strip():
                return ChangeValidation(
                    is_valid=False,
                    is_enhancement=False,
                    preserved_content=False,
                    added_content=[],
                    removed_content=[],
                    warnings=[],
                    error_message="New content is empty"
                )
            
            # Split content into lines for comparison
            old_lines = old_content.splitlines()
            new_lines = new_content.splitlines()
            
            # Calculate differences
            differ = difflib.unified_diff(old_lines, new_lines, lineterm='')
            diff_lines = list(differ)
            
            added_content = []
            removed_content = []
            
            for line in diff_lines:
                if line.startswith('+') and not line.startswith('+++'):
                    added_content.append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    removed_content.append(line[1:])
            
            # Check for content preservation
            preserved_content = len(removed_content) == 0 or self._is_safe_removal(removed_content)
            
            # Check if this is an enhancement (adds more than it removes)
            is_enhancement = len(added_content) > len(removed_content) * 0.5
            
            # Generate warnings
            warnings = []
            if len(removed_content) > 0:
                warnings.append(f"Content removal detected: {len(removed_content)} lines")
            
            if len(new_content) < len(old_content) * 0.8:
                warnings.append("Significant content reduction detected")
            
            # Check for critical sections
            critical_sections = self._check_critical_sections(old_content, new_content)
            if critical_sections:
                warnings.extend(critical_sections)
            
            # Overall validation
            is_valid = preserved_content and (is_enhancement or len(removed_content) == 0)
            
            return ChangeValidation(
                is_valid=is_valid,
                is_enhancement=is_enhancement,
                preserved_content=preserved_content,
                added_content=added_content,
                removed_content=removed_content,
                warnings=warnings
            )
            
        except Exception as e:
            return ChangeValidation(
                is_valid=False,
                is_enhancement=False,
                preserved_content=False,
                added_content=[],
                removed_content=[],
                warnings=[],
                error_message=f"Change validation failed: {str(e)}"
            )
    
    def _is_safe_removal(self, removed_content: List[str]) -> bool:
        """Check if removed content is safe (whitespace, comments, etc.)"""
        for line in removed_content:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Check if it's just formatting or empty lines
                if line and line not in ['', '---', '***']:
                    return False
        return True
    
    def _check_critical_sections(self, old_content: str, new_content: str) -> List[str]:
        """Check for modifications to critical sections"""
        warnings = []
        
        # Define critical section markers for feature documentation
        critical_markers = [
            "## Core Signal Features",
            "### Q50 (Primary Signal)",
            "## Risk & Volatility Features",
            "## 🎲 Position Sizing Features",
            "## Regime & Market Features"
        ]
        
        for marker in critical_markers:
            if marker in old_content and marker not in new_content:
                warnings.append(f"Critical section removed: {marker}")
            elif marker not in old_content and marker in new_content:
                warnings.append(f"New critical section added: {marker}")
        
        return warnings
    
    def enable_version_tracking(self, doc_path: str) -> bool:
        """
        Enable detailed version tracking for document
        
        Args:
            doc_path: Path to document to track
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_path = Path(doc_path)
            
            if not doc_path.exists():
                raise DocumentProtectionError(f"Document does not exist: {doc_path}")
            
            # Add to protected files list
            if str(doc_path) not in self.metadata["protected_files"]:
                self.metadata["protected_files"].append(str(doc_path))
            
            # Create initial version entry
            if str(doc_path) not in self.metadata["versions"]:
                self.metadata["versions"][str(doc_path)] = {
                    "enabled_at": datetime.now().isoformat(),
                    "current_hash": self._calculate_file_hash(str(doc_path)),
                    "version_history": []
                }
            
            self._save_metadata()
            return True
            
        except Exception as e:
            print(f"Failed to enable version tracking: {e}")
            return False
    
    def create_rollback_point(self, doc_path: str, description: str = "") -> RollbackPoint:
        """
        Create rollback point before major changes
        
        Args:
            doc_path: Path to document
            description: Description of the rollback point
            
        Returns:
            RollbackPoint object
        """
        try:
            doc_path = Path(doc_path)
            
            if not doc_path.exists():
                raise RollbackError(f"Document does not exist: {doc_path}")
            
            # Generate unique rollback point ID
            import time
            timestamp = self._generate_timestamp()
            # Add microseconds to ensure uniqueness
            unique_suffix = str(int(time.time() * 1000000))[-6:]
            rollback_id = f"{doc_path.stem}_{timestamp}_{unique_suffix}"
            
            # Create backup for rollback
            backup_result = self.create_backup(str(doc_path), f"Rollback point: {description}")
            
            if not backup_result.success:
                raise RollbackError(f"Failed to create backup: {backup_result.error_message}")
            
            # Create rollback point
            rollback_point = RollbackPoint(
                id=rollback_id,
                timestamp=timestamp,
                file_path=str(doc_path),
                backup_path=backup_result.backup_path,
                description=description,
                content_hash=backup_result.original_hash
            )
            
            # Save rollback point metadata
            self.metadata["rollback_points"][rollback_id] = asdict(rollback_point)
            self._save_metadata()
            
            return rollback_point
            
        except Exception as e:
            raise RollbackError(f"Failed to create rollback point: {str(e)}")
    
    def rollback_to_point(self, rollback_id: str) -> bool:
        """
        Rollback document to a specific rollback point
        
        Args:
            rollback_id: ID of the rollback point
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if rollback_id not in self.metadata["rollback_points"]:
                raise RollbackError(f"Rollback point not found: {rollback_id}")
            
            rollback_data = self.metadata["rollback_points"][rollback_id]
            rollback_point = RollbackPoint(**rollback_data)
            
            # Verify backup file exists
            if not Path(rollback_point.backup_path).exists():
                raise RollbackError(f"Backup file not found: {rollback_point.backup_path}")
            
            # Create backup of current state before rollback
            current_backup = self.create_backup(
                rollback_point.file_path, 
                f"Pre-rollback backup for {rollback_id}"
            )
            
            if not current_backup.success:
                raise RollbackError(f"Failed to backup current state: {current_backup.error_message}")
            
            # Perform rollback
            shutil.copy2(rollback_point.backup_path, rollback_point.file_path)
            
            # Verify rollback integrity - check that file was actually restored
            restored_hash = self._calculate_file_hash(rollback_point.file_path)
            expected_hash = rollback_point.content_hash
            if restored_hash != expected_hash:
                # Try to read backup content and compare
                with open(rollback_point.backup_path, 'rb') as f:
                    backup_content = f.read()
                backup_hash = hashlib.sha256(backup_content).hexdigest()
                
                if restored_hash != backup_hash:
                    raise RollbackError(f"Rollback integrity check failed. Expected: {expected_hash}, Got: {restored_hash}, Backup: {backup_hash}")
            
            # Update metadata
            rollback_info = {
                "rollback_id": rollback_id,
                "rollback_timestamp": datetime.now().isoformat(),
                "pre_rollback_backup": current_backup.backup_path
            }
            
            if "rollback_history" not in self.metadata:
                self.metadata["rollback_history"] = []
            
            self.metadata["rollback_history"].append(rollback_info)
            self._save_metadata()
            
            return True
            
        except RollbackError:
            raise  # Re-raise RollbackError for proper exception handling
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False
    
    def list_backups(self, doc_path: str = "") -> List[Dict[str, Any]]:
        """
        List all backups for a document or all documents
        
        Args:
            doc_path: Optional path to specific document
            
        Returns:
            List of backup information
        """
        if doc_path:
            return self.metadata["backups"].get(doc_path, [])
        else:
            all_backups = []
            for path, backups in self.metadata["backups"].items():
                all_backups.extend(backups)
            return sorted(all_backups, key=lambda x: x["created_at"], reverse=True)
    
    def list_rollback_points(self, doc_path: str = "") -> List[RollbackPoint]:
        """
        List all rollback points for a document or all documents
        
        Args:
            doc_path: Optional path to specific document
            
        Returns:
            List of RollbackPoint objects
        """
        rollback_points = []
        
        for rollback_id, data in self.metadata["rollback_points"].items():
            rollback_point = RollbackPoint(**data)
            if not doc_path or rollback_point.file_path == doc_path:
                rollback_points.append(rollback_point)
        
        return sorted(rollback_points, key=lambda x: x.timestamp, reverse=True)
    
    def cleanup_old_backups(self, days_to_keep: int = 30) -> int:
        """
        Clean up backups older than specified days
        
        Args:
            days_to_keep: Number of days to keep backups
            
        Returns:
            Number of backups cleaned up
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        for doc_path, backups in self.metadata["backups"].items():
            backups_to_keep = []
            
            for backup in backups:
                backup_date = datetime.fromisoformat(backup["created_at"])
                if backup_date >= cutoff_date:
                    backups_to_keep.append(backup)
                else:
                    # Remove backup file
                    backup_path = Path(backup["backup_path"])
                    if backup_path.exists():
                        backup_path.unlink()
                        cleaned_count += 1
            
            self.metadata["backups"][doc_path] = backups_to_keep
        
        self._save_metadata()
        return cleaned_count
    
    def get_protection_status(self, doc_path: str) -> Dict[str, Any]:
        """
        Get protection status for a document
        
        Args:
            doc_path: Path to document
            
        Returns:
            Dictionary with protection status information
        """
        doc_path = str(Path(doc_path))
        
        return {
            "is_protected": doc_path in self.metadata["protected_files"],
            "backup_count": len(self.metadata["backups"].get(doc_path, [])),
            "rollback_points": len([
                rp for rp in self.metadata["rollback_points"].values() 
                if rp["file_path"] == doc_path
            ]),
            "version_tracking": doc_path in self.metadata["versions"],
            "last_backup": self._get_last_backup_time(doc_path),
            "current_hash": self._calculate_file_hash(doc_path) if Path(doc_path).exists() else None
        }
    
    def _get_last_backup_time(self, doc_path: str) -> Optional[str]:
        """Get timestamp of last backup for document"""
        backups = self.metadata["backups"].get(doc_path, [])
        if backups:
            return max(backup["created_at"] for backup in backups)
        return None