DROP DATABASE IF EXISTS candidate_poll;
CREATE DATABASE candidate_poll;
ALTER SESSION SET CURRENT_DATABASE = 'candidate_poll';

CREATE TABLE "candidate" (
"Candidate_ID" int,
"People_ID" int,
"Poll_Source" text,
"Date" text,
"Support_rate" FLOAT,
"Consider_rate" FLOAT,
"Oppose_rate" FLOAT,
"Unsure_rate" FLOAT);

CREATE TABLE "people" (
"People_ID" int,
"Sex" text,
"Name" text,
"Date_of_Birth" text,
"Height" FLOAT,
"Weight" FLOAT);

INSERT INTO  "people" VALUES (1,'M','Hubert Henno','1976-10-06','188','83');
INSERT INTO  "people" VALUES (2,'M','Dominique Daquin','1972-11-10','197','85');
INSERT INTO  "people" VALUES (3,'F','Stéphane Antiga','1976-02-03','200','94');
INSERT INTO  "people" VALUES (4,'M','Laurent Capet','1972-05-05','202','92');
INSERT INTO  "people" VALUES (5,'F','Frantz Granvorka','1976-03-10','195','90');
INSERT INTO  "people" VALUES (6,'M','Vincent Montméat','1977-09-01','196','88');
INSERT INTO  "people" VALUES (7,'M','Loïc De Kergret','1970-08-20','193','89');
INSERT INTO  "people" VALUES (8,'M','Philippe Barça-Cysique','1977-04-22','194','88');
INSERT INTO  "people" VALUES (9,'M','Guillaume Samica','1981-09-28','196','82');

INSERT INTO  "candidate" VALUES (1,1,'WNBC/Marist Poll','Feb 12–15, 2007','0.25','0.30','0.43','0.2');
INSERT INTO  "candidate" VALUES (2,3,'WNBC/Marist Poll','Feb 12–15, 2007','0.17','0.42','0.32','0.9');
INSERT INTO  "candidate" VALUES (3,4,'FOX News/Opinion Dynamics Poll','Feb 13–14, 2007','0.18','0.34','0.44','0.3');
INSERT INTO  "candidate" VALUES (4,6,'Newsweek Poll','Nov 9–10, 2006','0.33','0.20','0.45','0.2');
INSERT INTO  "candidate" VALUES (5,7,'Newsweek Poll','Nov 9–10, 2006','0.24','0.30','0.32','0.4');
INSERT INTO  "candidate" VALUES (6,9,'Newsweek Poll','Nov 9–10, 2006','0.24','0.27','0.43','0.2');

