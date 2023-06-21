DROP DATABASE IF EXISTS soccer_2;
CREATE DATABASE soccer_2;
ALTER SESSION SET CURRENT_DATABASE = 'soccer_2';

DROP TABLE  IF EXISTS Player;
DROP TABLE  IF EXISTS Tryout;
DROP TABLE  IF EXISTS College;

CREATE TABLE 	College 
  ( cName   	TEXT NOT NULL,
    state   	TEXT,
    enr     	numeric(5,0));

CREATE TABLE 	Player
  ( pID			numeric(5,0) NOT NULL,
  	pName   	TEXT,
    yCard   	TEXT,
    HS      	numeric(5,0));

CREATE TABLE 	Tryout
  ( pID			numeric(5,0),
  	cName   	TEXT,
    pPos    	TEXT,
    decision    TEXT);


INSERT INTO College VALUES ('LSU', 'LA', 18000);
INSERT INTO College VALUES ('ASU', 'AZ', 12000);
INSERT INTO College VALUES ('OU', 'OK', 22000);
INSERT INTO College VALUES ('FSU', 'FL', 19000);

INSERT INTO Player VALUES (10001, 'Andrew', 'no', 1200);
INSERT INTO Player VALUES (20002, 'Blake', 'no', 1600);
INSERT INTO Player VALUES (30003, 'Charles', 'no', 300);
INSERT INTO Player VALUES (40004, 'David', 'yes', 1600);
INSERT INTO Player VALUES (40002, 'Drago', 'yes', 1600);
INSERT INTO Player VALUES (50005, 'Eddie', 'yes', 600);

INSERT INTO Tryout VALUES (10001, 'LSU', 'goalie', 'no');
INSERT INTO Tryout VALUES (10001, 'ASU', 'goalie', 'yes');
INSERT INTO Tryout VALUES (20002, 'FSU', 'striker', 'yes');
INSERT INTO Tryout VALUES (30003, 'OU', 'mid', 'no');
INSERT INTO Tryout VALUES (40004, 'ASU', 'goalie', 'no');
INSERT INTO Tryout VALUES (50005, 'LSU', 'mid', 'no');
