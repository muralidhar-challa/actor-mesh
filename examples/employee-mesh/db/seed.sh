#!/bin/sh
# db/seed.sh — create and seed employee SQLite database
# Run once: make db-seed

DB="${EMPLOYEE_DB:-/tmp/employee-mesh/db/employees.db}"
mkdir -p "$(dirname $DB)"

sqlite3 "$DB" << 'SQL'
CREATE TABLE IF NOT EXISTS employees (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL,
    role        TEXT NOT NULL,
    salary      INTEGER NOT NULL,
    hire_date   TEXT NOT NULL
);

DELETE FROM employees;

INSERT INTO employees (name, department, role, salary, hire_date) VALUES
    ('Alice Johnson',   'Engineering',  'Senior Engineer',      120000, '2019-03-15'),
    ('Bob Smith',       'Engineering',  'Junior Engineer',       75000, '2022-07-01'),
    ('Carol White',     'Engineering',  'Tech Lead',            150000, '2017-11-20'),
    ('David Brown',     'HR',           'HR Manager',            90000, '2018-05-10'),
    ('Eve Davis',       'HR',           'HR Coordinator',        65000, '2021-09-14'),
    ('Frank Miller',    'Sales',        'Sales Director',       130000, '2016-02-28'),
    ('Grace Wilson',    'Sales',        'Account Executive',     80000, '2020-06-03'),
    ('Henry Moore',     'Finance',      'CFO',                  200000, '2015-01-05'),
    ('Iris Taylor',     'Finance',      'Financial Analyst',     85000, '2021-03-22'),
    ('Jack Anderson',   'Marketing',    'Marketing Manager',     95000, '2019-08-17'),
    ('Karen Thomas',    'Marketing',    'Content Strategist',    70000, '2022-01-11'),
    ('Leo Jackson',     'Engineering',  'DevOps Engineer',      110000, '2020-04-30'),
    ('Mia Harris',      'Design',       'Lead Designer',        105000, '2018-12-01'),
    ('Noah Martin',     'Design',       'UX Designer',           78000, '2021-11-08'),
    ('Olivia Garcia',   'Engineering',  'Data Engineer',        115000, '2019-07-25');

SQL

echo "[db] seeded $(sqlite3 $DB 'SELECT COUNT(*) FROM employees') employees → $DB"
