DROP DATABASE IF EXISTS decoration_competition;
CREATE DATABASE decoration_competition;
ALTER SESSION SET CURRENT_DATABASE = 'decoration_competition';

CREATE TABLE "college" (
"College_ID" int,
"Name" text,
"Leader_Name" text,
"College_Location" text);



INSERT INTO  "college" VALUES ('1','Saskatchewan School','Ousame Tounkara','Ottawa');
INSERT INTO  "college" VALUES ('2','B.C. School','Ryan Thelwell','Minnesota');
INSERT INTO  "college" VALUES ('3','Calgary School','Andre Arlain','St. Francis Xavier');
INSERT INTO  "college" VALUES ('4','Edmonton School','Samir Chahine','McGill');
INSERT INTO  "college" VALUES ('5','Toronto School','Roger Dunbrack','Western Ontario');


CREATE TABLE "member_" (
"Member_ID" int,
"Name" text,
"Country" text,
"College_ID" int);


INSERT INTO  "member_" VALUES ('1','Jack Nicklaus','United States',1);
INSERT INTO  "member_" VALUES ('2','Billy Casper','United States',1);
INSERT INTO  "member_" VALUES ('3','Arnold Palmer','Canada',4);
INSERT INTO  "member_" VALUES ('4','Tom Watson','United States',4);
INSERT INTO  "member_" VALUES ('5','Homero Blancas','United States',2);
INSERT INTO  "member_" VALUES ('6','Pat Fitzsimons','Canada',5);
INSERT INTO  "member_" VALUES ('7','Bobby Nichols','Canada',5);
INSERT INTO  "member_" VALUES ('8','J. C. Snead','Canada',4);
INSERT INTO  "member_" VALUES ('9','Lee Trevino','United States',3);
INSERT INTO  "member_" VALUES ('10','Tom Weiskopf','United States',3);


CREATE TABLE "round" (
"Round_ID" int,
"Member_ID" int,
"Decoration_Theme" text,
"Rank_in_Round" int);


INSERT INTO  "round" VALUES (1,1,'Walk on the Moon',1);
INSERT INTO  "round" VALUES (1,2,'Soft Dream',2);
INSERT INTO  "round" VALUES (1,10,'Dark Nights',4);
INSERT INTO  "round" VALUES (2,4,'Sweetie',3);
INSERT INTO  "round" VALUES (2,6,'Summer',2);
INSERT INTO  "round" VALUES (2,9,'Happiness',1);

