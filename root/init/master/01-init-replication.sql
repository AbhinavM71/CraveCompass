-- root/init/db/01-master-init.sql

-- 1) Your existing table/schema creation and data load goes here...
--    (e.g., CREATE TABLE restaurants …; COPY restaurants FROM …; etc.)

-- 2) Enable replication on the master
ALTER SYSTEM SET wal_level = replica;
ALTER SYSTEM SET max_wal_senders = 5;
ALTER SYSTEM SET wal_keep_size = '64MB';

-- 3) Create the replication user (same password as your main user)
CREATE ROLE zomato WITH LOGIN REPLICATION PASSWORD 'zomato123';
