"""
Logging utilities for the feature test coverage system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the test coverage system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional custom log format string
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    logger = logging.getLogger("feature_test_coverage")
    logger.handlers.clear()  # Remove any existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(getattr(logging, log_level.upper()))
    return logger


def get_logger(name: str = "feature_test_coverage") -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TestExecutionLogger:
    """
    Specialized logger for test execution tracking.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize test execution logger.
        
        Args:
            log_file: Optional path to execution log file
        """
        self.logger = get_logger("test_execution")
        self.start_time = None
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
        
        if log_file:
            # Add file handler for execution logs
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - EXECUTION - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def start_test_suite(self, suite_name: str, test_count: int) -> None:
        """
        Log the start of a test suite.
        
        Args:
            suite_name: Name of the test suite
            test_count: Number of tests in the suite
        """
        self.start_time = datetime.now()
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
        
        self.logger.info(f"Starting test suite: {suite_name}")
        self.logger.info(f"Total tests to execute: {test_count}")
    
    def log_test_start(self, test_name: str, feature_name: str) -> None:
        """
        Log the start of an individual test.
        
        Args:
            test_name: Name of the test
            feature_name: Name of the feature being tested
        """
        self.logger.info(f"Starting test: {test_name} for feature: {feature_name}")
    
    def log_test_result(self, test_name: str, passed: bool, execution_time: float, message: str = "") -> None:
        """
        Log the result of an individual test.
        
        Args:
            test_name: Name of the test
            passed: Whether the test passed
            execution_time: Test execution time in seconds
            message: Optional additional message
        """
        self.test_count += 1
        if passed:
            self.passed_count += 1
            status = "PASSED"
        else:
            self.failed_count += 1
            status = "FAILED"
        
        log_message = f"Test {status}: {test_name} [{execution_time:.2f}s]"
        if message:
            log_message += f" - {message}"
        
        if passed:
            self.logger.info(log_message)
        else:
            self.logger.warning(log_message)
    
    def finish_test_suite(self) -> None:
        """Log the completion of a test suite."""
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
            success_rate = (self.passed_count / self.test_count) * 100 if self.test_count > 0 else 0
            
            self.logger.info("Test suite completed")
            self.logger.info(f"Total execution time: {total_time:.2f}s")
            self.logger.info(f"Tests executed: {self.test_count}")
            self.logger.info(f"Tests passed: {self.passed_count}")
            self.logger.info(f"Tests failed: {self.failed_count}")
            self.logger.info(f"Success rate: {success_rate:.1f}%")
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None) -> None:
        """
        Log an error during test execution.
        
        Args:
            error_message: Error description
            exception: Optional exception object
        """
        if exception:
            self.logger.error(f"{error_message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(error_message)