DROP DATABASE IF EXISTS performance_attendance;
CREATE DATABASE performance_attendance;
ALTER SESSION SET CURRENT_DATABASE = 'performance_attendance';


CREATE TABLE "member_" (
    "Member_ID" int,
    "Name" text,
    "Nationality" text,
    "Role" text);



INSERT INTO  "member_" VALUES (1,'Wilfredo Ruiz','Uruguay','Prime Violin');
INSERT INTO  "member_" VALUES (2,'Luis González','Argentina','Violin');
INSERT INTO  "member_" VALUES (3,'Héctor Campana','Argentina','Violin');
INSERT INTO  "member_" VALUES (4,'Héctor Campana','Argentina','Violin');
INSERT INTO  "member_" VALUES (5,'Juan Espil','Argentina','Viola');
INSERT INTO  "member_" VALUES (6,'Andrew Moten','United States','Viola');
INSERT INTO  "member_" VALUES (7,'Charlie Burke','United States','Viola');
INSERT INTO  "member_" VALUES (8,'Corey Allen','United States','Cello');
INSERT INTO  "member_" VALUES (9,'John Eubanks','United States','Cello');
INSERT INTO  "member_" VALUES (10,'Joshua Pittman','United States','Bass');
INSERT INTO  "member_" VALUES (11,'Joseph Bunn','United States','Bass');

CREATE TABLE "performance" (
    "Performance_ID" FLOAT,
    "Date" text,
    "Host" text,
    "Location" text,
    "Attendance" int);

INSERT INTO  "performance" VALUES (1,'February 2','Boston Bruins','TD Garden','165');
INSERT INTO  "performance" VALUES (2,'February 4','New York Rangers','Madison Square Garden','1820');
INSERT INTO  "performance" VALUES (3,'February 5','Atlanta Thrashers','Verizon Center','1878');
INSERT INTO  "performance" VALUES (4,'February 7','Pittsburgh Penguins','Verizon Center','1877');
INSERT INTO  "performance" VALUES (5,'February 10','Montreal Canadiens','Bell Centre','2173');
INSERT INTO  "performance" VALUES (6,'February 11','Ottawa Senators','Scotiabank Place','1982');


CREATE TABLE "member_attendance" (
    "Member_ID" int,
    "Performance_ID" int,
    "Num_of_Pieces" int);

INSERT INTO  "member_attendance" VALUES (11,3,2);
INSERT INTO  "member_attendance" VALUES (7,2,3);
INSERT INTO  "member_attendance" VALUES (4,6,4);
INSERT INTO  "member_attendance" VALUES (2,1,1);
INSERT INTO  "member_attendance" VALUES (3,1,1);
INSERT INTO  "member_attendance" VALUES (4,3,2);
INSERT INTO  "member_attendance" VALUES (5,1,2);
INSERT INTO  "member_attendance" VALUES (1,4,4);

