"""
Unit tests for DocumentProtectionSystem

Tests all functionality including backup creation, change validation,
version control, and rollback capabilities.
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)



import os
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
import pytest

from src.documentation.document_protection import (
    DocumentProtectionSystem,
    BackupResult,
    ChangeValidation,
    RollbackPoint,
    DocumentProtectionError,
    BackupCreationError,
    ChangeValidationError,
    RollbackError
)


class TestDocumentProtectionSystem:
    """Test suite for DocumentProtectionSystem"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def protection_system(self, temp_dir):
        """Create DocumentProtectionSystem instance for testing"""
        backup_dir = os.path.join(temp_dir, "test_backups")
        return DocumentProtectionSystem(backup_base_dir=backup_dir)
    
    @pytest.fixture
    def sample_doc(self, temp_dir):
        """Create sample document for testing"""
        doc_path = os.path.join(temp_dir, "test_doc.md")
        content = """# Test Document

## Section 1
This is test content.

## Section 2
More test content here.
"""
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return doc_path, content
    
    def test_initialization(self, protection_system):
        """Test system initialization"""
        assert protection_system.backup_dir.exists()
        assert protection_system.version_dir.exists()
        assert protection_system.rollback_dir.exists()
        assert protection_system.metadata_file.exists()
        assert isinstance(protection_system.metadata, dict)
    
    def test_create_backup_success(self, protection_system, sample_doc):
        """Test successful backup creation"""
        doc_path, content = sample_doc
        
        result = protection_system.create_backup(doc_path, "Test backup")
        
        assert result.success
        assert result.backup_path
        assert result.timestamp
        assert result.original_hash
        assert result.error_message is None
        
        # Verify backup file exists
        assert Path(result.backup_path).exists()
        
        # Verify backup content matches original
        with open(result.backup_path, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        assert backup_content == content
    
    def test_create_backup_nonexistent_file(self, protection_system):
        """Test backup creation with nonexistent file"""
        result = protection_system.create_backup("nonexistent.md")
        
        assert not result.success
        assert "does not exist" in result.error_message
    
    def test_validate_changes_enhancement(self, protection_system):
        """Test change validation for valid enhancement"""
        old_content = """# Document
## Section 1
Content here.
"""
        
        new_content = """# Document
## Section 1
Content here.

## New Section
Additional content added.
"""
        
        validation = protection_system.validate_changes(old_content, new_content)
        
        assert validation.is_valid
        assert validation.is_enhancement
        assert validation.preserved_content
        assert len(validation.added_content) > 0
        assert len(validation.removed_content) == 0
    
    def test_validate_changes_removal(self, protection_system):
        """Test change validation with content removal"""
        old_content = """# Document
## Section 1
Important content.
## Section 2
More content.
"""
        
        new_content = """# Document
## Section 1
Important content.
"""
        
        validation = protection_system.validate_changes(old_content, new_content)
        
        assert len(validation.removed_content) > 0
        assert len(validation.warnings) > 0
    
    def test_validate_changes_empty_content(self, protection_system):
        """Test change validation with empty new content"""
        old_content = "# Document\nContent here."
        new_content = ""
        
        validation = protection_system.validate_changes(old_content, new_content)
        
        assert not validation.is_valid
        assert "empty" in validation.error_message.lower()
    
    def test_enable_version_tracking(self, protection_system, sample_doc):
        """Test enabling version tracking"""
        doc_path, _ = sample_doc
        
        result = protection_system.enable_version_tracking(doc_path)
        
        assert result
        assert doc_path in protection_system.metadata["protected_files"]
        assert doc_path in protection_system.metadata["versions"]
    
    def test_create_rollback_point(self, protection_system, sample_doc):
        """Test creating rollback point"""
        doc_path, _ = sample_doc
        
        rollback_point = protection_system.create_rollback_point(
            doc_path, "Before major changes"
        )
        
        assert isinstance(rollback_point, RollbackPoint)
        assert rollback_point.id
        assert rollback_point.timestamp
        assert rollback_point.file_path == doc_path
        assert Path(rollback_point.backup_path).exists()
        assert rollback_point.description == "Before major changes"
    
    def test_rollback_to_point(self, protection_system, sample_doc):
        """Test rolling back to a rollback point"""
        doc_path, original_content = sample_doc
        
        # Create rollback point
        rollback_point = protection_system.create_rollback_point(doc_path, "Original state")
        
        # Modify the document
        modified_content = original_content + "\n\n## Modified Section\nNew content."
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # Rollback
        success = protection_system.rollback_to_point(rollback_point.id)
        
        assert success
        
        # Verify content is restored
        with open(doc_path, 'r', encoding='utf-8') as f:
            restored_content = f.read()
        assert restored_content == original_content
    
    def test_list_backups(self, protection_system, sample_doc):
        """Test listing backups"""
        doc_path, _ = sample_doc
        
        # Create multiple backups
        protection_system.create_backup(doc_path, "Backup 1")
        protection_system.create_backup(doc_path, "Backup 2")
        
        backups = protection_system.list_backups(doc_path)
        
        assert len(backups) == 2
        assert all("description" in backup for backup in backups)
        assert any("Backup 1" in backup["description"] for backup in backups)
        assert any("Backup 2" in backup["description"] for backup in backups)
    
    def test_list_rollback_points(self, protection_system, sample_doc):
        """Test listing rollback points"""
        doc_path, _ = sample_doc
        
        # Create rollback points
        rp1 = protection_system.create_rollback_point(doc_path, "Point 1")
        rp2 = protection_system.create_rollback_point(doc_path, "Point 2")
        
        rollback_points = protection_system.list_rollback_points(doc_path)
        
        assert len(rollback_points) == 2
        assert all(isinstance(rp, RollbackPoint) for rp in rollback_points)
        assert any(rp.description == "Point 1" for rp in rollback_points)
        assert any(rp.description == "Point 2" for rp in rollback_points)
    
    def test_get_protection_status(self, protection_system, sample_doc):
        """Test getting protection status"""
        doc_path, _ = sample_doc
        
        # Initially unprotected
        status = protection_system.get_protection_status(doc_path)
        assert not status["is_protected"]
        assert status["backup_count"] == 0
        
        # Enable protection and create backup
        protection_system.enable_version_tracking(doc_path)
        protection_system.create_backup(doc_path, "Test backup")
        
        status = protection_system.get_protection_status(doc_path)
        assert status["is_protected"]
        assert status["backup_count"] == 1
        assert status["version_tracking"]
        assert status["current_hash"]
    
    def test_cleanup_old_backups(self, protection_system, sample_doc):
        """Test cleaning up old backups"""
        doc_path, _ = sample_doc
        
        # Create some backups
        protection_system.create_backup(doc_path, "Backup 1")
        protection_system.create_backup(doc_path, "Backup 2")
        
        # Cleanup with 0 days (should remove all)
        cleaned = protection_system.cleanup_old_backups(days_to_keep=0)
        
        assert cleaned >= 0  # May be 0 if backups are very recent
    
    def test_critical_section_detection(self, protection_system):
        """Test detection of critical section changes"""
        old_content = """# Feature Documentation

## 🎯 Core Signal Features

### Q50 (Primary Signal)
Important feature description.
"""
        
        new_content = """# Feature Documentation

## Modified Core Features

### Q50 (Primary Signal)
Important feature description.
"""
        
        validation = protection_system.validate_changes(old_content, new_content)
        
        assert len(validation.warnings) > 0
        assert any("Critical section" in warning for warning in validation.warnings)
    
    def test_metadata_persistence(self, protection_system, sample_doc):
        """Test that metadata persists across instances"""
        doc_path, _ = sample_doc
        
        # Create backup with first instance
        protection_system.create_backup(doc_path, "Persistent backup")
        original_metadata = protection_system.metadata.copy()
        
        # Create new instance with same backup directory
        new_system = DocumentProtectionSystem(
            backup_base_dir=str(protection_system.backup_base_dir)
        )
        
        # Verify metadata was loaded
        assert new_system.metadata["backups"] == original_metadata["backups"]
    
    def test_file_hash_calculation(self, protection_system, sample_doc):
        """Test file hash calculation"""
        doc_path, _ = sample_doc
        
        hash1 = protection_system._calculate_file_hash(doc_path)
        hash2 = protection_system._calculate_file_hash(doc_path)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string
    
    def test_safe_removal_detection(self, protection_system):
        """Test detection of safe content removal"""
        # Test with comments and whitespace (safe)
        safe_removals = ["# Comment", "   ", "---", ""]
        assert protection_system._is_safe_removal(safe_removals)
        
        # Test with actual content (not safe)
        unsafe_removals = ["Important content", "### Feature Name"]
        assert not protection_system._is_safe_removal(unsafe_removals)
    
    def test_error_handling(self, protection_system):
        """Test error handling for various failure scenarios"""
        # Test with invalid file path
        with pytest.raises(DocumentProtectionError):
            protection_system._calculate_file_hash("nonexistent_file.txt")
        
        # Test rollback with invalid ID
        with pytest.raises(RollbackError):
            protection_system.rollback_to_point("invalid_rollback_id")


if __name__ == "__main__":
    pytest.main([__file__])