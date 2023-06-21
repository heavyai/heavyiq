DROP DATABASE IF EXISTS network_2;
CREATE DATABASE network_2;
ALTER SESSION SET CURRENT_DATABASE = 'network_2';

CREATE TABLE Person (
  name TEXT ,
  age INTEGER,
  city TEXT,
  gender TEXT,
  job TEXT
);

CREATE TABLE PersonFriend (
  name TEXT,
  friend TEXT,
  "year" INTEGER);

INSERT INTO Person VALUES ('Alice',25,'new york city','female','student');
INSERT INTO Person VALUES ('Bob',35,'salt lake city','male','engineer');
INSERT INTO Person VALUES ('Zach', 45,'austin','male','doctor');
 INSERT INTO Person VALUES ('Dan',26,'chicago','female','student');

INSERT INTO PersonFriend VALUES ('Alice','Bob',10);
INSERT INTO PersonFriend VALUES ('Zach','Dan', 12);
INSERT INTO PersonFriend VALUES ('Bob','Zach', 5);
INSERT INTO PersonFriend VALUES ('Zach','Alice', 6);
