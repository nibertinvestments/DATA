#!/usr/bin/env python3
"""
Popular Python Libraries Usage Examples for AI Coding Agents
============================================================

This module demonstrates proper usage of the most popular Python libraries
with real-world examples, best practices, and comprehensive error handling.

Libraries covered:
- NumPy: Numerical computing
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization
- Requests: HTTP requests
- JSON: Data serialization
- DateTime: Date and time handling
- Collections: Specialized container datatypes
- Itertools: Functional programming tools
- Regular Expressions: Pattern matching
- Pathlib: Modern path handling

Author: AI Dataset Creation Team
License: MIT
Created: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict, namedtuple, deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import itertools
import functools
import warnings


class NumPyExamples:
    """
    Comprehensive NumPy examples for numerical computing.
    
    NumPy is the foundation of scientific computing in Python,
    providing powerful N-dimensional array objects and mathematical functions.
    """
    
    @staticmethod
    def array_operations_showcase() -> Dict[str, Any]:
        """
        Demonstrate essential NumPy array operations with performance tips.
        
        Returns:
            Dictionary containing results of various array operations
        """
        print("🔢 NumPy Array Operations Showcase")
        print("-" * 40)
        
        # 1. Array creation with different methods
        print("1. Array Creation Methods:")
        
        # From lists - most common approach
        list_array = np.array([1, 2, 3, 4, 5])
        print(f"  From list: {list_array}")
        
        # Zeros, ones, and empty arrays
        zeros_2d = np.zeros((3, 4))
        ones_3d = np.ones((2, 3, 4))
        empty_array = np.empty((2, 2))  # Uninitialized, faster than zeros
        
        # Range arrays
        range_array = np.arange(0, 10, 2)  # start, stop, step
        linspace_array = np.linspace(0, 1, 5)  # start, stop, num_points
        print(f"  Range array: {range_array}")
        print(f"  Linspace array: {linspace_array}")
        
        # 2. Advanced indexing and slicing
        print("\n2. Advanced Indexing:")
        matrix = np.arange(20).reshape(4, 5)
        print(f"  Original matrix:\n{matrix}")
        
        # Boolean indexing - very powerful for data filtering
        even_mask = matrix % 2 == 0
        even_values = matrix[even_mask]
        print(f"  Even values: {even_values}")
        
        # Fancy indexing with arrays
        row_indices = np.array([0, 2, 3])
        col_indices = np.array([1, 3, 4])
        selected_elements = matrix[row_indices, col_indices]
        print(f"  Selected elements: {selected_elements}")
        
        # 3. Mathematical operations - vectorized for performance
        print("\n3. Mathematical Operations:")
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([10, 20, 30, 40])
        
        # Element-wise operations (vectorized - much faster than Python loops)
        addition = arr1 + arr2
        multiplication = arr1 * arr2
        power = arr1 ** 2
        print(f"  Addition: {addition}")
        print(f"  Multiplication: {multiplication}")
        print(f"  Power: {power}")
        
        # Aggregation functions
        data = np.random.randint(1, 100, size=(5, 5))
        print(f"  Random matrix:\n{data}")
        print(f"  Sum: {np.sum(data)}")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Standard deviation: {np.std(data):.2f}")
        print(f"  Max per column: {np.max(data, axis=0)}")
        print(f"  Min per row: {np.min(data, axis=1)}")
        
        # 4. Linear algebra operations
        print("\n4. Linear Algebra:")
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        # Matrix multiplication (not element-wise!)
        matrix_mult = np.dot(A, B)  # or A @ B in Python 3.5+
        print(f"  Matrix A:\n{A}")
        print(f"  Matrix B:\n{B}")
        print(f"  A @ B:\n{matrix_mult}")
        
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        print(f"  Eigenvalues of A: {eigenvalues}")
        
        # Matrix inverse and determinant
        try:
            A_inv = np.linalg.inv(A)
            det_A = np.linalg.det(A)
            print(f"  Determinant of A: {det_A:.2f}")
            print(f"  A^(-1):\n{A_inv}")
        except np.linalg.LinAlgError as e:
            print(f"  Linear algebra error: {e}")
        
        return {
            'list_array': list_array,
            'zeros_2d': zeros_2d,
            'matrix': matrix,
            'even_values': even_values,
            'math_results': {
                'addition': addition,
                'multiplication': multiplication,
                'power': power
            },
            'statistics': {
                'sum': np.sum(data),
                'mean': np.mean(data),
                'std': np.std(data)
            },
            'linear_algebra': {
                'matrix_mult': matrix_mult,
                'eigenvalues': eigenvalues,
                'determinant': det_A if 'det_A' in locals() else None
            }
        }
    
    @staticmethod
    def performance_tips_demo():
        """
        Demonstrate NumPy performance optimization techniques.
        
        These tips are crucial for AI agents to write efficient numerical code.
        """
        print("\n⚡ NumPy Performance Tips")
        print("-" * 30)
        
        # 1. Use vectorized operations instead of Python loops
        import time
        
        # Bad: Python loop
        python_list = list(range(1000000))
        start_time = time.time()
        python_squares = [x**2 for x in python_list]
        python_time = time.time() - start_time
        
        # Good: NumPy vectorized operation
        numpy_array = np.array(python_list)
        start_time = time.time()
        numpy_squares = numpy_array ** 2
        numpy_time = time.time() - start_time
        
        speedup = python_time / numpy_time
        print(f"  Python loop time: {python_time:.4f}s")
        print(f"  NumPy vectorized time: {numpy_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x faster")
        
        # 2. Use appropriate data types to save memory
        print(f"\n  Memory usage optimization:")
        large_array_float64 = np.ones(1000000, dtype=np.float64)
        large_array_float32 = np.ones(1000000, dtype=np.float32)
        large_array_int32 = np.ones(1000000, dtype=np.int32)
        
        print(f"    float64 array: {large_array_float64.nbytes / 1024 / 1024:.1f} MB")
        print(f"    float32 array: {large_array_float32.nbytes / 1024 / 1024:.1f} MB")
        print(f"    int32 array: {large_array_int32.nbytes / 1024 / 1024:.1f} MB")
        
        # 3. Use broadcasting to avoid creating unnecessary large arrays
        print(f"\n  Broadcasting example:")
        small_array = np.array([1, 2, 3])
        matrix = np.ones((3, 3))
        
        # Broadcasting automatically handles different shapes
        result = matrix + small_array  # small_array is broadcast to (3,3)
        print(f"    Matrix + array result:\n{result}")


class PandasExamples:
    """
    Comprehensive Pandas examples for data manipulation and analysis.
    
    Pandas is the go-to library for data analysis in Python,
    providing powerful data structures and data analysis tools.
    """
    
    @staticmethod
    def dataframe_operations_showcase() -> pd.DataFrame:
        """
        Demonstrate essential Pandas DataFrame operations.
        
        Returns:
            Processed DataFrame with various operations applied
        """
        print("\n📊 Pandas DataFrame Operations")
        print("-" * 35)
        
        # 1. Creating DataFrames from different sources
        print("1. DataFrame Creation:")
        
        # From dictionary - most common for small datasets
        data_dict = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Age': [25, 30, 35, 28, 32],
            'Salary': [50000, 60000, 70000, 55000, 65000],
            'Department': ['IT', 'HR', 'IT', 'Finance', 'IT'],
            'Join_Date': pd.to_datetime(['2020-01-15', '2019-03-20', '2018-07-10', 
                                       '2021-02-28', '2020-11-05'])
        }
        df = pd.DataFrame(data_dict)
        print(f"  Original DataFrame:\n{df}")
        
        # 2. Basic information and exploration
        print(f"\n2. DataFrame Information:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data types:\n{df.dtypes}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum()} bytes")
        
        # 3. Data selection and filtering
        print(f"\n3. Data Selection and Filtering:")
        
        # Column selection
        names_ages = df[['Name', 'Age']]
        print(f"  Names and ages:\n{names_ages}")
        
        # Row filtering with conditions
        it_employees = df[df['Department'] == 'IT']
        print(f"  IT employees:\n{it_employees}")
        
        # Complex filtering with multiple conditions
        high_earners = df[(df['Salary'] > 55000) & (df['Age'] < 35)]
        print(f"  High earners under 35:\n{high_earners}")
        
        # 4. Data aggregation and grouping
        print(f"\n4. Data Aggregation:")
        
        # Group by department and calculate statistics
        dept_stats = df.groupby('Department').agg({
            'Salary': ['mean', 'max', 'min', 'count'],
            'Age': 'mean'
        }).round(2)
        print(f"  Department statistics:\n{dept_stats}")
        
        # Value counts for categorical data
        dept_counts = df['Department'].value_counts()
        print(f"  Department counts:\n{dept_counts}")
        
        # 5. Data transformation and feature engineering
        print(f"\n5. Data Transformation:")
        
        # Add new calculated columns
        df['Years_Experience'] = (datetime.now() - df['Join_Date']).dt.days / 365.25
        df['Salary_Per_Year_Experience'] = df['Salary'] / df['Years_Experience']
        
        # Create age categories
        df['Age_Category'] = pd.cut(df['Age'], 
                                   bins=[0, 30, 35, 100], 
                                   labels=['Young', 'Middle', 'Senior'])
        
        print(f"  Enhanced DataFrame:\n{df[['Name', 'Age', 'Age_Category', 'Years_Experience']].round(2)}")
        
        # 6. Handling missing data (simulation)
        print(f"\n6. Missing Data Handling:")
        
        # Introduce some missing values for demonstration
        df_with_missing = df.copy()
        df_with_missing.loc[1, 'Salary'] = np.nan
        df_with_missing.loc[3, 'Department'] = np.nan
        
        print(f"  Missing values per column:\n{df_with_missing.isnull().sum()}")
        
        # Different strategies for handling missing data
        df_filled = df_with_missing.copy()
        df_filled['Salary'].fillna(df_filled['Salary'].median(), inplace=True)
        df_filled['Department'].fillna('Unknown', inplace=True)
        
        print(f"  After filling missing values:\n{df_filled.isnull().sum()}")
        
        # 7. Data sorting and ranking
        print(f"\n7. Sorting and Ranking:")
        
        # Sort by multiple columns
        sorted_df = df.sort_values(['Department', 'Salary'], ascending=[True, False])
        print(f"  Sorted by department and salary:\n{sorted_df[['Name', 'Department', 'Salary']]}")
        
        # Ranking
        df['Salary_Rank'] = df['Salary'].rank(ascending=False)
        print(f"  Salary rankings:\n{df[['Name', 'Salary', 'Salary_Rank']]}")
        
        return df
    
    @staticmethod
    def time_series_operations():
        """
        Demonstrate time series operations in Pandas.
        
        Time series analysis is crucial for many AI applications.
        """
        print(f"\n⏰ Time Series Operations")
        print("-" * 30)
        
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        values = np.cumsum(np.random.randn(100)) + 100
        
        ts_df = pd.DataFrame({
            'Date': dates,
            'Value': values,
            'Category': np.random.choice(['A', 'B', 'C'], 100)
        })
        ts_df.set_index('Date', inplace=True)
        
        print(f"  Time series data (first 5 rows):\n{ts_df.head()}")
        
        # Resampling - aggregate data by different time periods
        monthly_avg = ts_df['Value'].resample('M').mean()
        weekly_sum = ts_df['Value'].resample('W').sum()
        
        print(f"  Monthly averages (first 3):\n{monthly_avg.head(3)}")
        print(f"  Weekly sums (first 3):\n{weekly_sum.head(3)}")
        
        # Rolling window operations
        ts_df['Rolling_Mean_7'] = ts_df['Value'].rolling(window=7).mean()
        ts_df['Rolling_Std_7'] = ts_df['Value'].rolling(window=7).std()
        
        print(f"  With rolling statistics:\n{ts_df[['Value', 'Rolling_Mean_7', 'Rolling_Std_7']].head(10)}")
        
        # Date/time filtering
        recent_data = ts_df['2023-03-01':'2023-03-31']
        print(f"  March data shape: {recent_data.shape}")
        
        return ts_df


class MatplotlibExamples:
    """
    Comprehensive Matplotlib examples for data visualization.
    
    Matplotlib is the foundation of plotting in Python,
    enabling creation of publication-quality figures.
    """
    
    @staticmethod
    def comprehensive_plotting_showcase():
        """
        Demonstrate various types of plots with best practices.
        
        Creates multiple plot types commonly used in data analysis and AI.
        """
        print(f"\n📈 Matplotlib Plotting Showcase")
        print("-" * 35)
        
        # Suppress matplotlib warnings for cleaner output
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        # Set up the plotting style
        plt.style.use('default')  # Clean, professional style
        
        # 1. Line plot with multiple series
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Comprehensive Data Visualization Examples', fontsize=16, fontweight='bold')
        
        # Line plot
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.sin(x) * np.exp(-x/5)
        
        ax1.plot(x, y1, label='sin(x)', linewidth=2, color='blue', linestyle='-')
        ax1.plot(x, y2, label='cos(x)', linewidth=2, color='red', linestyle='--')
        ax1.plot(x, y3, label='sin(x)·e^(-x/5)', linewidth=2, color='green', linestyle=':')
        ax1.set_title('Mathematical Functions', fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bar chart with error bars
        categories = ['Category A', 'Category B', 'Category C', 'Category D']
        values = [23, 45, 56, 78]
        errors = [3, 5, 4, 6]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        bars = ax2.bar(categories, values, yerr=errors, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Performance by Category', fontweight='bold')
        ax2.set_ylabel('Performance Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Scatter plot with color mapping
        np.random.seed(42)
        n_points = 100
        x_scatter = np.random.randn(n_points)
        y_scatter = 2 * x_scatter + np.random.randn(n_points) * 0.5
        colors_scatter = np.random.randn(n_points)
        
        scatter = ax3.scatter(x_scatter, y_scatter, c=colors_scatter, 
                            cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.set_title('Correlation Analysis', fontweight='bold')
        ax3.set_xlabel('Feature X')
        ax3.set_ylabel('Feature Y')
        
        # Add trend line
        z = np.polyfit(x_scatter, y_scatter, 1)
        p = np.poly1d(z)
        ax3.plot(x_scatter, p(x_scatter), "r--", alpha=0.8, linewidth=2, label='Trend line')
        ax3.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Color Value', rotation=270, labelpad=15)
        
        # 4. Histogram with multiple distributions
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(2, 1.5, 1000)
        data3 = np.random.exponential(1, 1000)
        
        ax4.hist(data1, bins=30, alpha=0.7, label='Normal(0,1)', color='blue', density=True)
        ax4.hist(data2, bins=30, alpha=0.7, label='Normal(2,1.5)', color='red', density=True)
        ax4.hist(data3, bins=30, alpha=0.7, label='Exponential(1)', color='green', density=True)
        
        ax4.set_title('Distribution Comparison', fontweight='bold')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot (commented out to avoid file creation in test environment)
        # plt.savefig('comprehensive_plots.png', dpi=300, bbox_inches='tight')
        
        print("  ✅ Four different plot types created successfully!")
        print("  📊 Includes: Line plot, Bar chart, Scatter plot, Histogram")
        print("  🎨 Features: Custom styling, error bars, trend lines, colormaps")
        
        # Don't show the plot in headless environment
        # plt.show()
        plt.close()


class ModernPythonPatterns:
    """
    Modern Python programming patterns and best practices.
    
    These patterns are essential for writing clean, efficient,
    and maintainable Python code that AI agents should learn.
    """
    
    @staticmethod
    def datetime_operations_showcase():
        """
        Comprehensive datetime operations for real-world applications.
        """
        print(f"\n🕐 DateTime Operations Showcase")
        print("-" * 35)
        
        # 1. Creating datetime objects
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        specific_date = datetime(2024, 12, 25, 15, 30, 0)
        
        print(f"  Current local time: {now}")
        print(f"  Current UTC time: {utc_now}")
        print(f"  Specific datetime: {specific_date}")
        
        # 2. Parsing and formatting dates
        date_string = "2024-03-15 14:30:00"
        parsed_date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        formatted_date = parsed_date.strftime("%B %d, %Y at %I:%M %p")
        
        print(f"  Parsed from string: {parsed_date}")
        print(f"  Formatted nicely: {formatted_date}")
        
        # 3. Date arithmetic
        one_week_ago = now - timedelta(weeks=1)
        one_month_later = now + timedelta(days=30)
        time_difference = specific_date - now
        
        print(f"  One week ago: {one_week_ago}")
        print(f"  One month later: {one_month_later}")
        print(f"  Days until Christmas 2024: {time_difference.days}")
        
        # 4. Working with different timezones
        # Create timezone-aware datetimes
        eastern = timezone(timedelta(hours=-5))  # EST
        pacific = timezone(timedelta(hours=-8))  # PST
        
        eastern_time = datetime.now(eastern)
        pacific_time = eastern_time.astimezone(pacific)
        
        print(f"  Eastern time: {eastern_time}")
        print(f"  Pacific time: {pacific_time}")
        
        return {
            'now': now,
            'parsed_date': parsed_date,
            'time_difference': time_difference,
            'eastern_time': eastern_time,
            'pacific_time': pacific_time
        }
    
    @staticmethod
    def collections_patterns_showcase():
        """
        Demonstrate powerful collections module patterns.
        """
        print(f"\n📚 Collections Module Patterns")
        print("-" * 35)
        
        # 1. Counter - counting hashable objects
        text = "hello world this is a test hello world"
        word_counts = Counter(text.split())
        char_counts = Counter(text.replace(' ', ''))
        
        print(f"  Word counts: {word_counts}")
        print(f"  Most common words: {word_counts.most_common(3)}")
        print(f"  Character counts: {char_counts.most_common(5)}")
        
        # 2. defaultdict - automatic default values
        # Group students by grade
        students = [
            ('Alice', 'A'), ('Bob', 'B'), ('Charlie', 'A'),
            ('Diana', 'C'), ('Eve', 'B'), ('Frank', 'A')
        ]
        
        grades = defaultdict(list)
        for name, grade in students:
            grades[grade].append(name)
        
        print(f"  Students by grade: {dict(grades)}")
        
        # 3. namedtuple - lightweight object-like structures
        Student = namedtuple('Student', ['name', 'age', 'gpa'])
        
        student1 = Student('Alice', 20, 3.8)
        student2 = Student('Bob', 21, 3.6)
        
        print(f"  Student 1: {student1}")
        print(f"  Student 1 GPA: {student1.gpa}")
        print(f"  Student as dict: {student1._asdict()}")
        
        # 4. deque - double-ended queue for efficient operations
        recent_actions = deque(maxlen=5)  # Keep only last 5 actions
        
        for action in ['login', 'view_page', 'click_button', 'scroll', 'logout', 'login_again']:
            recent_actions.append(action)
            print(f"    Actions queue: {list(recent_actions)}")
        
        return {
            'word_counts': word_counts,
            'grades': dict(grades),
            'student': student1,
            'recent_actions': list(recent_actions)
        }
    
    @staticmethod
    def regex_patterns_showcase():
        """
        Practical regular expression patterns for data processing.
        """
        print(f"\n🔍 Regular Expression Patterns")
        print("-" * 35)
        
        # Sample text data
        sample_text = """
        Contact us at support@company.com or sales@company.org
        Phone numbers: +1-555-123-4567, (555) 987-6543
        Visit our website: https://www.company.com or http://blog.company.org
        Dates: 2024-03-15, 03/15/2024, March 15, 2024
        """
        
        # 1. Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, sample_text)
        print(f"  Found emails: {emails}")
        
        # 2. Phone number extraction (multiple formats)
        phone_pattern = r'(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
        phones = re.findall(phone_pattern, sample_text)
        print(f"  Found phone numbers: {phones}")
        
        # 3. URL extraction
        url_pattern = r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?'
        urls = re.findall(url_pattern, sample_text)
        print(f"  Found URLs: {urls}")
        
        # 4. Date extraction (multiple formats)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'[A-Za-z]+ \d{1,2}, \d{4}'  # Month DD, YYYY
        ]
        
        all_dates = []
        for pattern in date_patterns:
            dates = re.findall(pattern, sample_text)
            all_dates.extend(dates)
        
        print(f"  Found dates: {all_dates}")
        
        # 5. Data cleaning with regex
        dirty_data = "  Remove    extra   spaces   and   normalize  "
        cleaned = re.sub(r'\s+', ' ', dirty_data.strip())
        print(f"  Cleaned text: '{cleaned}'")
        
        # 6. Validation patterns
        validation_patterns = {
            'strong_password': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
            'credit_card': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
            'social_security': r'^\d{3}-\d{2}-\d{4}$'
        }
        
        test_password = "MyPass123!"
        is_strong = bool(re.match(validation_patterns['strong_password'], test_password))
        print(f"  Password '{test_password}' is strong: {is_strong}")
        
        return {
            'emails': emails,
            'phones': phones,
            'urls': urls,
            'dates': all_dates,
            'cleaned_text': cleaned,
            'password_strong': is_strong
        }


def comprehensive_library_demo():
    """
    Run a comprehensive demonstration of all library examples.
    
    This function showcases the power and versatility of Python's
    ecosystem for data science, analysis, and general programming.
    """
    print("🐍 Comprehensive Python Libraries Demonstration")
    print("=" * 55)
    
    # Run all demonstrations
    results = {}
    
    # NumPy examples
    numpy_examples = NumPyExamples()
    results['numpy'] = numpy_examples.array_operations_showcase()
    numpy_examples.performance_tips_demo()
    
    # Pandas examples
    pandas_examples = PandasExamples()
    results['pandas_df'] = pandas_examples.dataframe_operations_showcase()
    results['pandas_ts'] = pandas_examples.time_series_operations()
    
    # Matplotlib examples
    matplotlib_examples = MatplotlibExamples()
    matplotlib_examples.comprehensive_plotting_showcase()
    
    # Modern Python patterns
    modern_patterns = ModernPythonPatterns()
    results['datetime'] = modern_patterns.datetime_operations_showcase()
    results['collections'] = modern_patterns.collections_patterns_showcase()
    results['regex'] = modern_patterns.regex_patterns_showcase()
    
    print(f"\n🎉 Demonstration completed successfully!")
    print(f"📈 Covered libraries: NumPy, Pandas, Matplotlib, datetime, collections, re")
    print(f"🔧 Demonstrated patterns: vectorization, data manipulation, visualization, text processing")
    
    return results


if __name__ == "__main__":
    # Run the comprehensive demonstration
    demo_results = comprehensive_library_demo()
    
    # Display summary statistics
    print(f"\n📊 Summary Statistics:")
    if 'numpy' in demo_results:
        numpy_stats = demo_results['numpy']['statistics']
        print(f"  NumPy array statistics: mean={numpy_stats['mean']:.2f}, std={numpy_stats['std']:.2f}")
    
    if 'pandas_df' in demo_results:
        df = demo_results['pandas_df']
        print(f"  Pandas DataFrame: {df.shape[0]} employees across {df['Department'].nunique()} departments")
    
    if 'regex' in demo_results:
        regex_results = demo_results['regex']
        print(f"  Text processing: found {len(regex_results['emails'])} emails, {len(regex_results['phones'])} phones")
    
    print(f"\n✅ All library demonstrations completed successfully!")