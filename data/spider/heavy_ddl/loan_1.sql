DROP DATABASE IF EXISTS loan_1;
CREATE DATABASE loan_1;
ALTER SESSION SET CURRENT_DATABASE = 'loan_1';

CREATE TABLE bank (
branch_ID int ,
bname TEXT,
no_of_customers int,
city TEXT,
state TEXT);


CREATE TABLE customer (
cust_ID TEXT ,
cust_name TEXT,
acc_type text,
acc_bal int,
no_of_loans int,
credit_score int,
branch_ID int,
state TEXT);


CREATE TABLE loan (
loan_ID TEXT ,
loan_type TEXT,
cust_ID TEXT,
branch_ID TEXT,
amount int);

insert into bank values (1, 'morningside', 203, 'New York City', 'New York');
insert into bank values (2, 'downtown', 123, 'Salt Lake City', 'Utah');
insert into bank values (3, 'broadway', 453, 'New York City', 'New York');
insert into bank values (4, 'high', 367, 'Austin', 'Texas');

insert into customer values (1, 'Mary', 'saving', 2000, 2, 30, 2, 'Utah');
insert into customer values (2, 'Jack', 'checking', 1000, 1, 20, 1, 'Texas');
insert into customer values (3, 'Owen', 'saving', 800000, 0, 210, 3, 'New York');

insert into loan values (1, 'Mortgages', 1, 1, 2050);
insert into loan values (2, 'Auto', 1, 2, 3000);
insert into loan values (3, 'Business', 3, 3, 5000);
