DROP DATABASE IF EXISTS body_builder;
CREATE DATABASE body_builder;
ALTER SESSION SET CURRENT_DATABASE = 'body_builder';

CREATE TABLE "body_builder" (
"Body_Builder_ID" int,
"People_ID" int,
"Snatch" FLOAT,
"Clean_Jerk" FLOAT,
"Total" FLOAT);

CREATE TABLE "people" (
"People_ID" int,
"Name" text,
"Height" FLOAT,
"Weight" FLOAT,
"Birth_Date" text,
"Birth_Place" text);

INSERT INTO  "people" VALUES (1,'Jack Campbell','182','80','January 1, 1992','Port Huron, Michigan');
INSERT INTO  "people" VALUES (2,'Ty Conklin','192','90','March 30, 1976','Anchorage, Alaska');
INSERT INTO  "people" VALUES (3,'Al Montoya','195','100','February 13, 1985','Glenview, Illinois');
INSERT INTO  "people" VALUES (4,'Mark Fayne','215','102','May 5, 1987','Nashua, New Hampshire');
INSERT INTO  "people" VALUES (5,'Cam Fowler','196','89','December 5, 1991','Farmington Hills, Michigan');
INSERT INTO  "people" VALUES (6,'Jake Gardiner','205','92','July 4, 1990','Minnetonka, Minnesota');


INSERT INTO  "body_builder" VALUES (1,1,'142.5','175.0','317.5');
INSERT INTO  "body_builder" VALUES (2,2,'137.5','177.5','315.0');
INSERT INTO  "body_builder" VALUES (3,3,'140.0','175.0','315.0');
INSERT INTO  "body_builder" VALUES (4,5,'137.5','175.0','312.5');
INSERT INTO  "body_builder" VALUES (5,6,'130.0','162.5','292.5');


