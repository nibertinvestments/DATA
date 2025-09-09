-- Comprehensive SQL examples for database operations and data analysis

-- =====================================================
-- SQL Programming Examples for ML/AI Training
-- Covers: DDL, DML, DQL, Advanced Queries, Functions
-- =====================================================

-- Data Definition Language (DDL) - Creating Database Structure
-- ============================================================

-- Create database (MySQL/PostgreSQL syntax)
-- CREATE DATABASE company_db;
-- USE company_db;

-- Create tables with various data types and constraints
CREATE TABLE departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    department_name VARCHAR(100) NOT NULL UNIQUE,
    location VARCHAR(100),
    budget DECIMAL(15,2) DEFAULT 0,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    manager_id INT
);

CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    hire_date DATE NOT NULL,
    salary DECIMAL(10,2) NOT NULL CHECK (salary > 0),
    department_id INT,
    manager_id INT,
    status ENUM('active', 'inactive', 'terminated') DEFAULT 'active',
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id),
    INDEX idx_department (department_id),
    INDEX idx_salary (salary),
    INDEX idx_hire_date (hire_date)
);

CREATE TABLE projects (
    project_id INT PRIMARY KEY AUTO_INCREMENT,
    project_name VARCHAR(200) NOT NULL,
    description TEXT,
    start_date DATE NOT NULL,
    end_date DATE,
    budget DECIMAL(15,2),
    status ENUM('planning', 'active', 'completed', 'cancelled') DEFAULT 'planning',
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    CHECK (end_date IS NULL OR end_date >= start_date)
);

CREATE TABLE employee_projects (
    employee_id INT,
    project_id INT,
    role VARCHAR(100),
    hours_allocated DECIMAL(5,2) DEFAULT 0,
    start_date DATE,
    end_date DATE,
    PRIMARY KEY (employee_id, project_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE TABLE salary_history (
    history_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT NOT NULL,
    old_salary DECIMAL(10,2),
    new_salary DECIMAL(10,2),
    change_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason VARCHAR(200),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

-- Create view for employee details
CREATE VIEW employee_details AS
SELECT 
    e.employee_id,
    CONCAT(e.first_name, ' ', e.last_name) AS full_name,
    e.email,
    e.salary,
    d.department_name,
    d.location,
    DATEDIFF(CURRENT_DATE, e.hire_date) AS days_employed,
    YEAR(CURRENT_DATE) - YEAR(e.hire_date) AS years_employed
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id
WHERE e.status = 'active';

-- Data Manipulation Language (DML) - Inserting Data
-- =================================================

-- Insert departments
INSERT INTO departments (department_name, location, budget, manager_id) VALUES
('Information Technology', 'New York', 500000.00, NULL),
('Human Resources', 'Chicago', 200000.00, NULL),
('Finance', 'New York', 300000.00, NULL),
('Marketing', 'Los Angeles', 250000.00, NULL),
('Research & Development', 'San Francisco', 800000.00, NULL),
('Sales', 'Boston', 400000.00, NULL);

-- Insert employees
INSERT INTO employees (first_name, last_name, email, phone, hire_date, salary, department_id, manager_id) VALUES
('John', 'Smith', 'john.smith@company.com', '555-0101', '2020-01-15', 85000.00, 1, NULL),
('Sarah', 'Johnson', 'sarah.johnson@company.com', '555-0102', '2019-03-22', 92000.00, 1, 1),
('Michael', 'Brown', 'michael.brown@company.com', '555-0103', '2021-06-10', 78000.00, 1, 1),
('Emily', 'Davis', 'emily.davis@company.com', '555-0104', '2018-11-05', 95000.00, 2, NULL),
('David', 'Wilson', 'david.wilson@company.com', '555-0105', '2020-08-18', 88000.00, 3, NULL),
('Lisa', 'Anderson', 'lisa.anderson@company.com', '555-0106', '2019-12-02', 91000.00, 4, NULL),
('Robert', 'Taylor', 'robert.taylor@company.com', '555-0107', '2021-02-14', 83000.00, 5, NULL),
('Jennifer', 'Martinez', 'jennifer.martinez@company.com', '555-0108', '2020-05-30', 87000.00, 6, NULL),
('William', 'Garcia', 'william.garcia@company.com', '555-0109', '2019-07-12', 79000.00, 1, 2),
('Amanda', 'Rodriguez', 'amanda.rodriguez@company.com', '555-0110', '2021-09-08', 81000.00, 3, 5);

-- Update manager assignments
UPDATE departments SET manager_id = 1 WHERE department_name = 'Information Technology';
UPDATE departments SET manager_id = 4 WHERE department_name = 'Human Resources';
UPDATE departments SET manager_id = 5 WHERE department_name = 'Finance';
UPDATE departments SET manager_id = 6 WHERE department_name = 'Marketing';
UPDATE departments SET manager_id = 7 WHERE department_name = 'Research & Development';
UPDATE departments SET manager_id = 8 WHERE department_name = 'Sales';

-- Insert projects
INSERT INTO projects (project_name, description, start_date, end_date, budget, status, department_id) VALUES
('CRM System Upgrade', 'Upgrade customer relationship management system', '2023-01-01', '2023-06-30', 150000.00, 'active', 1),
('Employee Portal', 'Develop internal employee portal', '2023-02-15', '2023-08-15', 120000.00, 'active', 1),
('Marketing Campaign Q2', 'Spring marketing campaign', '2023-04-01', '2023-06-30', 80000.00, 'planning', 4),
('Financial Audit 2023', 'Annual financial audit process', '2023-03-01', '2023-05-31', 50000.00, 'active', 3),
('New Product Research', 'Research for next generation product', '2023-01-15', '2023-12-31', 200000.00, 'active', 5),
('Sales Training Program', 'Comprehensive sales training initiative', '2023-05-01', '2023-07-31', 75000.00, 'planning', 6);

-- Assign employees to projects
INSERT INTO employee_projects (employee_id, project_id, role, hours_allocated, start_date) VALUES
(1, 1, 'Project Manager', 40.00, '2023-01-01'),
(2, 1, 'Senior Developer', 35.00, '2023-01-01'),
(3, 1, 'Developer', 30.00, '2023-01-15'),
(1, 2, 'Technical Lead', 20.00, '2023-02-15'),
(9, 2, 'Developer', 40.00, '2023-02-15'),
(6, 3, 'Campaign Manager', 40.00, '2023-04-01'),
(5, 4, 'Financial Analyst', 25.00, '2023-03-01'),
(10, 4, 'Audit Assistant', 20.00, '2023-03-01'),
(7, 5, 'Research Lead', 40.00, '2023-01-15'),
(8, 6, 'Training Coordinator', 30.00, '2023-05-01');

-- Data Query Language (DQL) - Basic Queries
-- =========================================

-- Basic SELECT statements
SELECT * FROM employees;

SELECT first_name, last_name, salary 
FROM employees 
WHERE salary > 85000;

SELECT DISTINCT department_id 
FROM employees;

-- Conditional queries
SELECT first_name, last_name, salary,
    CASE 
        WHEN salary >= 90000 THEN 'High'
        WHEN salary >= 80000 THEN 'Medium'
        ELSE 'Low'
    END AS salary_grade
FROM employees
ORDER BY salary DESC;

-- Pattern matching
SELECT * FROM employees 
WHERE email LIKE '%@company.com';

SELECT * FROM employees 
WHERE first_name LIKE 'J%';

-- Date functions and filtering
SELECT first_name, last_name, hire_date,
    DATEDIFF(CURRENT_DATE, hire_date) AS days_employed,
    YEAR(CURRENT_DATE) - YEAR(hire_date) AS years_of_service
FROM employees
WHERE hire_date >= '2020-01-01';

-- Advanced Queries - JOINs
-- ========================

-- INNER JOIN
SELECT e.first_name, e.last_name, d.department_name, d.location
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- LEFT JOIN with aggregation
SELECT d.department_name, 
    COUNT(e.employee_id) AS employee_count,
    AVG(e.salary) AS avg_salary,
    MAX(e.salary) AS max_salary,
    MIN(e.salary) AS min_salary
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_id, d.department_name
ORDER BY employee_count DESC;

-- Multiple JOINs
SELECT e.first_name, e.last_name, p.project_name, ep.role, ep.hours_allocated
FROM employees e
INNER JOIN employee_projects ep ON e.employee_id = ep.employee_id
INNER JOIN projects p ON ep.project_id = p.project_id
WHERE p.status = 'active';

-- Self JOIN (employees and their managers)
SELECT 
    emp.first_name + ' ' + emp.last_name AS employee_name,
    mgr.first_name + ' ' + mgr.last_name AS manager_name
FROM employees emp
LEFT JOIN employees mgr ON emp.manager_id = mgr.employee_id;

-- Subqueries
-- ==========

-- Scalar subquery
SELECT first_name, last_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Correlated subquery
SELECT e1.first_name, e1.last_name, e1.salary, e1.department_id
FROM employees e1
WHERE e1.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department_id = e1.department_id
);

-- EXISTS subquery
SELECT d.department_name
FROM departments d
WHERE EXISTS (
    SELECT 1 
    FROM employees e 
    WHERE e.department_id = d.department_id
    AND e.salary > 85000
);

-- Window Functions (Advanced Analytics)
-- ===================================

-- Ranking functions
SELECT 
    first_name, 
    last_name, 
    salary,
    department_id,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as salary_rank,
    RANK() OVER (ORDER BY salary DESC) as overall_rank,
    DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;

-- Running totals and moving averages
SELECT 
    first_name,
    last_name,
    hire_date,
    salary,
    SUM(salary) OVER (ORDER BY hire_date) as running_total_salary,
    AVG(salary) OVER (ORDER BY hire_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as moving_avg_salary
FROM employees
ORDER BY hire_date;

-- Lag and Lead functions
SELECT 
    first_name,
    last_name,
    hire_date,
    salary,
    LAG(salary, 1) OVER (ORDER BY hire_date) as previous_hire_salary,
    LEAD(salary, 1) OVER (ORDER BY hire_date) as next_hire_salary,
    salary - LAG(salary, 1) OVER (ORDER BY hire_date) as salary_difference
FROM employees
ORDER BY hire_date;

-- Common Table Expressions (CTEs)
-- ===============================

-- Simple CTE
WITH high_earners AS (
    SELECT first_name, last_name, salary, department_id
    FROM employees
    WHERE salary > 85000
)
SELECT he.first_name, he.last_name, he.salary, d.department_name
FROM high_earners he
JOIN departments d ON he.department_id = d.department_id;

-- Recursive CTE (organizational hierarchy)
WITH RECURSIVE org_hierarchy AS (
    -- Base case: top-level managers
    SELECT employee_id, first_name, last_name, manager_id, 0 as level, 
           CAST(first_name + ' ' + last_name AS VARCHAR(500)) as hierarchy_path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees with managers
    SELECT e.employee_id, e.first_name, e.last_name, e.manager_id, oh.level + 1,
           CAST(oh.hierarchy_path + ' -> ' + e.first_name + ' ' + e.last_name AS VARCHAR(500))
    FROM employees e
    INNER JOIN org_hierarchy oh ON e.manager_id = oh.employee_id
)
SELECT * FROM org_hierarchy ORDER BY level, hierarchy_path;

-- Advanced Aggregations and Analytics
-- ===================================

-- Complex grouping with ROLLUP
SELECT 
    COALESCE(d.department_name, 'ALL DEPARTMENTS') as department,
    COALESCE(CAST(YEAR(e.hire_date) AS VARCHAR), 'ALL YEARS') as hire_year,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary,
    SUM(salary) as total_salary
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id
GROUP BY ROLLUP(d.department_name, YEAR(e.hire_date))
ORDER BY department, hire_year;

-- Pivot-like analysis using conditional aggregation
SELECT 
    d.department_name,
    COUNT(*) as total_employees,
    SUM(CASE WHEN e.salary >= 90000 THEN 1 ELSE 0 END) as high_salary_count,
    SUM(CASE WHEN e.salary BETWEEN 80000 AND 89999 THEN 1 ELSE 0 END) as medium_salary_count,
    SUM(CASE WHEN e.salary < 80000 THEN 1 ELSE 0 END) as low_salary_count,
    ROUND(AVG(e.salary), 2) as avg_salary
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_id, d.department_name
ORDER BY avg_salary DESC;

-- Stored Procedures and Functions
-- ===============================

-- Stored procedure for salary updates
DELIMITER $$
CREATE PROCEDURE UpdateEmployeeSalary(
    IN emp_id INT,
    IN new_salary DECIMAL(10,2),
    IN reason VARCHAR(200)
)
BEGIN
    DECLARE old_salary DECIMAL(10,2);
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;
    
    START TRANSACTION;
    
    -- Get current salary
    SELECT salary INTO old_salary FROM employees WHERE employee_id = emp_id;
    
    -- Update employee salary
    UPDATE employees SET salary = new_salary WHERE employee_id = emp_id;
    
    -- Insert into salary history
    INSERT INTO salary_history (employee_id, old_salary, new_salary, reason)
    VALUES (emp_id, old_salary, new_salary, reason);
    
    COMMIT;
END$$
DELIMITER ;

-- Function to calculate bonus
DELIMITER $$
CREATE FUNCTION CalculateBonus(emp_salary DECIMAL(10,2), performance_rating INT)
RETURNS DECIMAL(10,2)
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE bonus DECIMAL(10,2) DEFAULT 0;
    
    CASE performance_rating
        WHEN 5 THEN SET bonus = emp_salary * 0.15;  -- Excellent: 15%
        WHEN 4 THEN SET bonus = emp_salary * 0.10;  -- Good: 10%
        WHEN 3 THEN SET bonus = emp_salary * 0.05;  -- Satisfactory: 5%
        WHEN 2 THEN SET bonus = emp_salary * 0.02;  -- Needs Improvement: 2%
        ELSE SET bonus = 0;                         -- Unsatisfactory: 0%
    END CASE;
    
    RETURN bonus;
END$$
DELIMITER ;

-- Triggers
-- ========

-- Trigger to automatically update budget when employee salary changes
DELIMITER $$
CREATE TRIGGER update_department_budget_after_salary_change
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    IF OLD.salary != NEW.salary THEN
        UPDATE departments 
        SET budget = budget + (NEW.salary - OLD.salary) 
        WHERE department_id = NEW.department_id;
    END IF;
END$$
DELIMITER ;

-- Indexes for Performance Optimization
-- ===================================

-- Composite indexes for common query patterns
CREATE INDEX idx_employee_dept_salary ON employees(department_id, salary);
CREATE INDEX idx_employee_hire_date_status ON employees(hire_date, status);
CREATE INDEX idx_project_status_dates ON projects(status, start_date, end_date);

-- Full-text search index (if supported)
-- CREATE FULLTEXT INDEX idx_project_description ON projects(description);

-- Data Analysis Queries
-- =====================

-- Monthly hiring trends
SELECT 
    YEAR(hire_date) as hire_year,
    MONTH(hire_date) as hire_month,
    COUNT(*) as hires_count,
    AVG(salary) as avg_starting_salary
FROM employees
GROUP BY YEAR(hire_date), MONTH(hire_date)
ORDER BY hire_year, hire_month;

-- Department productivity analysis
SELECT 
    d.department_name,
    COUNT(DISTINCT e.employee_id) as employee_count,
    COUNT(DISTINCT p.project_id) as project_count,
    SUM(ep.hours_allocated) as total_allocated_hours,
    AVG(ep.hours_allocated) as avg_hours_per_assignment,
    ROUND(SUM(ep.hours_allocated) / COUNT(DISTINCT e.employee_id), 2) as hours_per_employee
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
LEFT JOIN employee_projects ep ON e.employee_id = ep.employee_id
LEFT JOIN projects p ON ep.project_id = p.project_id
GROUP BY d.department_id, d.department_name
ORDER BY hours_per_employee DESC;

-- Performance monitoring queries
-- =============================

-- Find employees without projects
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN employee_projects ep ON e.employee_id = ep.employee_id
LEFT JOIN departments d ON e.department_id = d.department_id
WHERE ep.employee_id IS NULL;

-- Project status summary
SELECT 
    status,
    COUNT(*) as project_count,
    SUM(budget) as total_budget,
    AVG(DATEDIFF(COALESCE(end_date, CURRENT_DATE), start_date)) as avg_duration_days
FROM projects
GROUP BY status
ORDER BY 
    CASE status
        WHEN 'active' THEN 1
        WHEN 'planning' THEN 2
        WHEN 'completed' THEN 3
        WHEN 'cancelled' THEN 4
    END;

-- Data cleanup and maintenance
-- ============================

-- Remove completed projects older than 2 years
-- DELETE FROM projects 
-- WHERE status = 'completed' 
-- AND end_date < DATE_SUB(CURRENT_DATE, INTERVAL 2 YEAR);

-- Archive old salary history (keep only last 5 years)
-- CREATE TABLE salary_history_archive AS 
-- SELECT * FROM salary_history 
-- WHERE change_date < DATE_SUB(CURRENT_DATE, INTERVAL 5 YEAR);

-- Security and Access Control
-- ===========================

-- Create users with different privilege levels
-- CREATE USER 'hr_manager'@'localhost' IDENTIFIED BY 'secure_password';
-- CREATE USER 'finance_analyst'@'localhost' IDENTIFIED BY 'secure_password';
-- CREATE USER 'project_manager'@'localhost' IDENTIFIED BY 'secure_password';

-- Grant appropriate permissions
-- GRANT SELECT, INSERT, UPDATE ON employees TO 'hr_manager'@'localhost';
-- GRANT SELECT ON salary_history TO 'hr_manager'@'localhost';
-- GRANT SELECT ON employee_details TO 'project_manager'@'localhost';
-- GRANT SELECT ON departments, projects TO 'finance_analyst'@'localhost';

-- This comprehensive SQL example covers:
-- 1. Database schema design with constraints and indexes
-- 2. Data insertion and manipulation
-- 3. Basic and advanced queries
-- 4. JOINs and subqueries
-- 5. Window functions and analytics
-- 6. Common Table Expressions (CTEs)
-- 7. Stored procedures and functions
-- 8. Triggers for automation
-- 9. Performance optimization
-- 10. Data analysis and reporting
-- 11. Security and access control

-- Perfect for training ML/AI models on SQL syntax, patterns, and best practices!