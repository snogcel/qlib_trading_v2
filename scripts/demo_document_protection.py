#!/usr/bin/env python3
"""
Demonstration script for DocumentProtectionSystem

This script demonstrates how to use the DocumentProtectionSystem to protect
the FEATURE_DOCUMENTATION.md file with backup, version control, and rollback capabilities.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from documentation.document_protection import DocumentProtectionSystem


def main():
    """Demonstrate DocumentProtectionSystem functionality"""
    
    print("🛡️  Document Protection System Demo")
    print("=" * 50)
    
    # Initialize protection system
    print("\n1. Initializing Document Protection System...")
    protection_system = DocumentProtectionSystem()
    
    # Target document
    doc_path = "docs/FEATURE_DOCUMENTATION.md"
    
    if not Path(doc_path).exists():
        print(f"Error: {doc_path} not found!")
        return
    
    print(f"Protection system initialized")
    print(f"   Backup directory: {protection_system.backup_dir}")
    print(f"   Version directory: {protection_system.version_dir}")
    print(f"   Rollback directory: {protection_system.rollback_dir}")
    
    # Check current protection status
    print(f"\n2. Checking protection status for {doc_path}...")
    status = protection_system.get_protection_status(doc_path)
    
    print(f"   Protected: {status['is_protected']}")
    print(f"   Backup count: {status['backup_count']}")
    print(f"   Rollback points: {status['rollback_points']}")
    print(f"   Version tracking: {status['version_tracking']}")
    print(f"   Current hash: {status['current_hash'][:16]}..." if status['current_hash'] else "   Current hash: None")
    
    # Enable version tracking
    print(f"\n3. Enabling version tracking...")
    if protection_system.enable_version_tracking(doc_path):
        print("Version tracking enabled")
    else:
        print("Failed to enable version tracking")
    
    # Create initial backup
    print(f"\n4. Creating initial backup...")
    backup_result = protection_system.create_backup(
        doc_path, 
        "Initial backup before feature documentation enhancement"
    )
    
    if backup_result.success:
        print("Backup created successfully")
        print(f"   Backup path: {backup_result.backup_path}")
        print(f"   Timestamp: {backup_result.timestamp}")
        print(f"   Original hash: {backup_result.original_hash[:16]}...")
    else:
        print(f"Backup failed: {backup_result.error_message}")
        return
    
    # Create rollback point
    print(f"\n5. Creating rollback point...")
    try:
        rollback_point = protection_system.create_rollback_point(
            doc_path,
            "Before thesis statement enhancement"
        )
        print("Rollback point created")
        print(f"   Rollback ID: {rollback_point.id}")
        print(f"   Description: {rollback_point.description}")
        print(f"   Backup path: {rollback_point.backup_path}")
    except Exception as e:
        print(f"Rollback point creation failed: {e}")
        return
    
    # Demonstrate change validation
    print(f"\n6. Demonstrating change validation...")
    
    # Read current content
    with open(doc_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Simulate enhancement (adding thesis statement)
    enhanced_content = original_content.replace(
        "### Q50 (Primary Signal)",
        """### Q50 (Primary Signal)
**Economic Thesis**: Q50 represents the 50th percentile probability threshold that captures the balance point between supply and demand forces in the market. When Q50 signals indicate directional bias, it suggests that the underlying probability distribution has shifted sufficiently to create exploitable inefficiencies."""
    )
    
    # Validate the change
    validation = protection_system.validate_changes(original_content, enhanced_content, doc_path)
    
    print(f"   Change validation results:")
    print(f"   - Valid: {validation.is_valid}")
    print(f"   - Is enhancement: {validation.is_enhancement}")
    print(f"   - Content preserved: {validation.preserved_content}")
    print(f"   - Added lines: {len(validation.added_content)}")
    print(f"   - Removed lines: {len(validation.removed_content)}")
    print(f"   - Warnings: {len(validation.warnings)}")
    
    if validation.warnings:
        for warning in validation.warnings:
            print(f"      {warning}")
    
    # List existing backups
    print(f"\n7. Listing existing backups...")
    backups = protection_system.list_backups(doc_path)
    
    if backups:
        print(f"   Found {len(backups)} backup(s):")
        for i, backup in enumerate(backups[:3], 1):  # Show first 3
            print(f"   {i}. {backup['timestamp']} - {backup['description']}")
    else:
        print("   No backups found")
    
    # List rollback points
    print(f"\n8. Listing rollback points...")
    rollback_points = protection_system.list_rollback_points(doc_path)
    
    if rollback_points:
        print(f"   Found {len(rollback_points)} rollback point(s):")
        for i, rp in enumerate(rollback_points[:3], 1):  # Show first 3
            print(f"   {i}. {rp.id} - {rp.description}")
    else:
        print("   No rollback points found")
    
    # Final status check
    print(f"\n9. Final protection status...")
    final_status = protection_system.get_protection_status(doc_path)
    
    print(f"   Protected: {final_status['is_protected']}")
    print(f"   Backup count: {final_status['backup_count']}")
    print(f"   Rollback points: {final_status['rollback_points']}")
    print(f"   Version tracking: {final_status['version_tracking']}")
    print(f"   Last backup: {final_status['last_backup']}")
    
    print(f"\nDocument Protection System Demo Complete!")
    print(f"\nThe FEATURE_DOCUMENTATION.md file is now protected with:")
    print(f"   - Automatic backups in docs/research/case_study/backups/")
    print(f"   - Version tracking enabled")
    print(f"   - Rollback points for safe recovery")
    print(f"   - Change validation to prevent data loss")
    
    print(f"\n📝 Next steps:")
    print(f"   1. Use protection_system.create_backup() before making changes")
    print(f"   2. Use protection_system.validate_changes() to verify enhancements")
    print(f"   3. Use protection_system.create_rollback_point() before major changes")
    print(f"   4. Use protection_system.rollback_to_point() if recovery is needed")


if __name__ == "__main__":
    main()