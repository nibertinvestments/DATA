"""
Comprehensive File I/O and Data Processing Examples in Python
Demonstrates various file formats, data processing, and error handling.
"""

import csv
import json
import pickle
import sqlite3
import os
import shutil
import gzip
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Union
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
from contextlib import contextmanager
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Employee:
    """Employee data class for demonstration."""
    id: int
    name: str
    department: str
    salary: float
    email: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Employee':
        """Create Employee from dictionary."""
        return cls(**data)


class FileProcessor:
    """Comprehensive file processing utility class."""
    
    def __init__(self, base_path: str = "/tmp"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def create_sample_data(self) -> List[Employee]:
        """Create sample employee data."""
        return [
            Employee(1, "Alice Johnson", "Engineering", 85000.0, "alice@company.com"),
            Employee(2, "Bob Smith", "Marketing", 65000.0, "bob@company.com"),
            Employee(3, "Carol Davis", "Engineering", 90000.0, "carol@company.com"),
            Employee(4, "David Wilson", "Sales", 70000.0, "david@company.com"),
            Employee(5, "Eve Brown", "HR", 60000.0, "eve@company.com"),
        ]
    
    def write_csv_file(self, employees: List[Employee], filename: str) -> str:
        """Write employee data to CSV file."""
        filepath = self.base_path / filename
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'name', 'department', 'salary', 'email']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for employee in employees:
                    writer.writerow(employee.to_dict())
            
            logger.info(f"CSV file written: {filepath}")
            return str(filepath)
            
        except IOError as e:
            logger.error(f"Failed to write CSV file: {e}")
            raise
    
    def read_csv_file(self, filename: str) -> List[Employee]:
        """Read employee data from CSV file."""
        filepath = self.base_path / filename
        employees = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # Convert salary to float and id to int
                    row['id'] = int(row['id'])
                    row['salary'] = float(row['salary'])
                    employees.append(Employee.from_dict(row))
            
            logger.info(f"Read {len(employees)} employees from CSV")
            return employees
            
        except (IOError, ValueError) as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise
    
    def write_json_file(self, employees: List[Employee], filename: str, pretty: bool = True) -> str:
        """Write employee data to JSON file."""
        filepath = self.base_path / filename
        
        try:
            data = [employee.to_dict() for employee in employees]
            
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                if pretty:
                    json.dump(data, jsonfile, indent=2, ensure_ascii=False)
                else:
                    json.dump(data, jsonfile, ensure_ascii=False)
            
            logger.info(f"JSON file written: {filepath}")
            return str(filepath)
            
        except IOError as e:
            logger.error(f"Failed to write JSON file: {e}")
            raise
    
    def read_json_file(self, filename: str) -> List[Employee]:
        """Read employee data from JSON file."""
        filepath = self.base_path / filename
        
        try:
            with open(filepath, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
            
            employees = [Employee.from_dict(emp_data) for emp_data in data]
            logger.info(f"Read {len(employees)} employees from JSON")
            return employees
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read JSON file: {e}")
            raise
    
    def write_xml_file(self, employees: List[Employee], filename: str) -> str:
        """Write employee data to XML file."""
        filepath = self.base_path / filename
        
        try:
            # Create root element
            root = ET.Element("employees")
            
            for employee in employees:
                emp_elem = ET.SubElement(root, "employee")
                emp_elem.set("id", str(employee.id))
                
                # Add child elements
                name_elem = ET.SubElement(emp_elem, "name")
                name_elem.text = employee.name
                
                dept_elem = ET.SubElement(emp_elem, "department")
                dept_elem.text = employee.department
                
                salary_elem = ET.SubElement(emp_elem, "salary")
                salary_elem.text = str(employee.salary)
                
                email_elem = ET.SubElement(emp_elem, "email")
                email_elem.text = employee.email
            
            # Write to file
            tree = ET.ElementTree(root)
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"XML file written: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to write XML file: {e}")
            raise
    
    def read_xml_file(self, filename: str) -> List[Employee]:
        """Read employee data from XML file."""
        filepath = self.base_path / filename
        employees = []
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            for emp_elem in root.findall("employee"):
                employee = Employee(
                    id=int(emp_elem.get("id")),
                    name=emp_elem.find("name").text,
                    department=emp_elem.find("department").text,
                    salary=float(emp_elem.find("salary").text),
                    email=emp_elem.find("email").text
                )
                employees.append(employee)
            
            logger.info(f"Read {len(employees)} employees from XML")
            return employees
            
        except (ET.ParseError, AttributeError, ValueError) as e:
            logger.error(f"Failed to read XML file: {e}")
            raise
    
    def write_pickle_file(self, employees: List[Employee], filename: str) -> str:
        """Write employee data to pickle file."""
        filepath = self.base_path / filename
        
        try:
            with open(filepath, 'wb') as picklefile:
                pickle.dump(employees, picklefile)
            
            logger.info(f"Pickle file written: {filepath}")
            return str(filepath)
            
        except IOError as e:
            logger.error(f"Failed to write pickle file: {e}")
            raise
    
    def read_pickle_file(self, filename: str) -> List[Employee]:
        """Read employee data from pickle file."""
        filepath = self.base_path / filename
        
        try:
            with open(filepath, 'rb') as picklefile:
                employees = pickle.load(picklefile)
            
            logger.info(f"Read {len(employees)} employees from pickle")
            return employees
            
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to read pickle file: {e}")
            raise


class DatabaseManager:
    """SQLite database manager for employee data."""
    
    def __init__(self, db_path: str = "/tmp/employees.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database and create tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    salary REAL NOT NULL,
                    email TEXT UNIQUE NOT NULL
                )
            ''')
            conn.commit()
    
    def insert_employees(self, employees: List[Employee]):
        """Insert employees into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for employee in employees:
                cursor.execute('''
                    INSERT OR REPLACE INTO employees 
                    (id, name, department, salary, email) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (employee.id, employee.name, employee.department, 
                     employee.salary, employee.email))
            
            conn.commit()
            logger.info(f"Inserted {len(employees)} employees into database")
    
    def get_employees_by_department(self, department: str) -> List[Employee]:
        """Get employees by department."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, name, department, salary, email FROM employees WHERE department = ?',
                (department,)
            )
            
            employees = []
            for row in cursor.fetchall():
                employees.append(Employee(*row))
            
            return employees
    
    def get_salary_statistics(self) -> Dict[str, float]:
        """Get salary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    AVG(salary) as avg_salary,
                    MIN(salary) as min_salary,
                    MAX(salary) as max_salary,
                    COUNT(*) as total_employees
                FROM employees
            ''')
            
            row = cursor.fetchone()
            return {
                'average_salary': row[0],
                'minimum_salary': row[1],
                'maximum_salary': row[2],
                'total_employees': row[3]
            }


class FileArchiver:
    """File compression and archiving utilities."""
    
    @staticmethod
    def compress_file_gzip(source_path: str, target_path: str = None) -> str:
        """Compress a file using gzip."""
        if target_path is None:
            target_path = source_path + '.gz'
        
        try:
            with open(source_path, 'rb') as f_in:
                with gzip.open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"File compressed: {target_path}")
            return target_path
            
        except IOError as e:
            logger.error(f"Failed to compress file: {e}")
            raise
    
    @staticmethod
    def decompress_file_gzip(source_path: str, target_path: str = None) -> str:
        """Decompress a gzip file."""
        if target_path is None:
            target_path = source_path.replace('.gz', '')
        
        try:
            with gzip.open(source_path, 'rb') as f_in:
                with open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"File decompressed: {target_path}")
            return target_path
            
        except IOError as e:
            logger.error(f"Failed to decompress file: {e}")
            raise
    
    @staticmethod
    def create_zip_archive(file_paths: List[str], archive_path: str) -> str:
        """Create a ZIP archive from multiple files."""
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        # Use just the filename in the archive
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)
                    else:
                        logger.warning(f"File not found: {file_path}")
            
            logger.info(f"ZIP archive created: {archive_path}")
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to create ZIP archive: {e}")
            raise
    
    @staticmethod
    def extract_zip_archive(archive_path: str, extract_dir: str) -> List[str]:
        """Extract files from a ZIP archive."""
        extracted_files = []
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_dir)
                extracted_files = [os.path.join(extract_dir, name) for name in zipf.namelist()]
            
            logger.info(f"ZIP archive extracted to: {extract_dir}")
            return extracted_files
            
        except Exception as e:
            logger.error(f"Failed to extract ZIP archive: {e}")
            raise


class FileUtils:
    """General file utility functions."""
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """Calculate hash of a file."""
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except IOError as e:
            logger.error(f"Failed to calculate hash: {e}")
            raise
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = path.stat()
        
        return {
            'name': path.name,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_timestamp': stat.st_ctime,
            'modified_timestamp': stat.st_mtime,
            'is_file': path.is_file(),
            'is_directory': path.is_dir(),
            'suffix': path.suffix,
            'parent_directory': str(path.parent),
            'absolute_path': str(path.absolute())
        }
    
    @staticmethod
    @contextmanager
    def temporary_directory():
        """Context manager for temporary directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @staticmethod
    def safe_file_write(file_path: str, content: str, backup: bool = True) -> bool:
        """Write file safely with optional backup."""
        path = Path(file_path)
        
        try:
            # Create backup if file exists
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.backup')
                shutil.copy2(path, backup_path)
                logger.info(f"Backup created: {backup_path}")
            
            # Write to temporary file first
            temp_path = path.with_suffix(path.suffix + '.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Atomic move
            temp_path.rename(path)
            logger.info(f"File written safely: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file safely: {e}")
            return False


def demonstrate_file_formats():
    """Demonstrate various file format operations."""
    processor = FileProcessor()
    employees = processor.create_sample_data()
    
    print("=== File Format Demonstrations ===")
    
    # CSV operations
    csv_file = processor.write_csv_file(employees, "employees.csv")
    csv_data = processor.read_csv_file("employees.csv")
    print(f"CSV: Wrote {len(employees)} employees, read {len(csv_data)} employees")
    
    # JSON operations
    json_file = processor.write_json_file(employees, "employees.json")
    json_data = processor.read_json_file("employees.json")
    print(f"JSON: Wrote {len(employees)} employees, read {len(json_data)} employees")
    
    # XML operations
    xml_file = processor.write_xml_file(employees, "employees.xml")
    xml_data = processor.read_xml_file("employees.xml")
    print(f"XML: Wrote {len(employees)} employees, read {len(xml_data)} employees")
    
    # Pickle operations
    pickle_file = processor.write_pickle_file(employees, "employees.pkl")
    pickle_data = processor.read_pickle_file("employees.pkl")
    print(f"Pickle: Wrote {len(employees)} employees, read {len(pickle_data)} employees")
    
    return [csv_file, json_file, xml_file, pickle_file]


def demonstrate_database_operations():
    """Demonstrate database operations."""
    db_manager = DatabaseManager("/tmp/demo_employees.db")
    processor = FileProcessor()
    employees = processor.create_sample_data()
    
    print("\n=== Database Operations ===")
    
    # Insert employees
    db_manager.insert_employees(employees)
    
    # Query by department
    eng_employees = db_manager.get_employees_by_department("Engineering")
    print(f"Engineering department has {len(eng_employees)} employees")
    
    # Get statistics
    stats = db_manager.get_salary_statistics()
    print(f"Salary statistics: Avg=${stats['average_salary']:.2f}, "
          f"Min=${stats['minimum_salary']:.2f}, Max=${stats['maximum_salary']:.2f}")


def demonstrate_file_archiving():
    """Demonstrate file compression and archiving."""
    print("\n=== File Archiving ===")
    
    # Create some sample files first
    file_paths = demonstrate_file_formats()
    
    # Compress a file with gzip
    gzip_file = FileArchiver.compress_file_gzip(file_paths[0])
    print(f"Compressed file created: {os.path.basename(gzip_file)}")
    
    # Create ZIP archive
    zip_file = "/tmp/employees_archive.zip"
    FileArchiver.create_zip_archive(file_paths, zip_file)
    print(f"ZIP archive created with {len(file_paths)} files")
    
    # Extract ZIP archive
    with FileUtils.temporary_directory() as temp_dir:
        extracted = FileArchiver.extract_zip_archive(zip_file, temp_dir)
        print(f"Extracted {len(extracted)} files to temporary directory")


def demonstrate_file_utilities():
    """Demonstrate file utility functions."""
    print("\n=== File Utilities ===")
    
    # Create a sample file
    sample_file = "/tmp/sample.txt"
    with open(sample_file, 'w') as f:
        f.write("This is a sample file for demonstration purposes.\n" * 100)
    
    # Get file information
    file_info = FileUtils.get_file_info(sample_file)
    print(f"File size: {file_info['size_mb']:.2f} MB")
    print(f"File suffix: {file_info['suffix']}")
    
    # Calculate file hash
    file_hash = FileUtils.calculate_file_hash(sample_file)
    print(f"File hash (SHA256): {file_hash[:16]}...")
    
    # Safe file write
    success = FileUtils.safe_file_write(sample_file, "New content with backup", backup=True)
    print(f"Safe file write: {'Success' if success else 'Failed'}")


if __name__ == "__main__":
    print("=== Comprehensive File I/O and Data Processing Examples ===\n")
    
    # Run all demonstrations
    demonstrate_file_formats()
    demonstrate_database_operations()
    demonstrate_file_archiving()
    demonstrate_file_utilities()
    
    print("\n=== Features Demonstrated ===")
    print("- CSV, JSON, XML, and Pickle file operations")
    print("- SQLite database operations")
    print("- File compression (gzip, ZIP)")
    print("- File utilities (hashing, info, safe write)")
    print("- Error handling and logging")
    print("- Context managers and temporary files")
    print("- Data classes and type hints")