DROP DATABASE IF EXISTS climbing;
CREATE DATABASE climbing;
ALTER SESSION SET CURRENT_DATABASE = 'climbing';

CREATE TABLE "mountain" (
"Mountain_ID" int,
"Name" text,
"Height" FLOAT,
"Prominence" FLOAT,
"Range" text,
"Country" text);

CREATE TABLE "climber" (
"Climber_ID" int,
"Name" text,
"Country" text,
"Time" text,
"Points" FLOAT,
"Mountain_ID" int);

INSERT INTO  "mountain" VALUES (1,'Kibo (Uhuru Pk)','5895','5885','Kilimanjaro','Tanzania');
INSERT INTO  "mountain" VALUES (2,'Mount Kenya (Batian)','5199','3825','Mount Kenya','Kenya');
INSERT INTO  "mountain" VALUES (3,'Mawenzi (Hans Meyer Pk)','5148','850','Kilimanjaro','Tanzania');
INSERT INTO  "mountain" VALUES (4,'Ngaliema / Mt Stanley (Margherita Pk)','5109','3951','Rwenzori','DR Congo Uganda');
INSERT INTO  "mountain" VALUES (5,'Mount Kenya (Lenana)','4985','130','Mount Kenya','Kenya');
INSERT INTO  "mountain" VALUES (6,'Ngaliema / Mt Stanley (Savoia Pk)','4977','110','Rwenzori','Uganda');
INSERT INTO  "mountain" VALUES (7,'Duwoni / Mt Speke (Vittorio Emanuele Pk)','4890','720','Rwenzori','Uganda');


INSERT INTO  "climber" VALUES ('1','Klaus Enders','West Germany','1:13.05.6','15',1);
INSERT INTO  "climber" VALUES ('2','Siegfried Schauzu','West Germany','1:14.56.4','12',1);
INSERT INTO  "climber" VALUES ('3','Hans Luthringhauser','West Germany','1:16.58.0','10',2);
INSERT INTO  "climber" VALUES ('4','Jean Claude Castella','Switzerland','1:17.16.0','8',2);
INSERT INTO  "climber" VALUES ('5','Horst Owesle','West Germany','1:17.22.0','6',2);
INSERT INTO  "climber" VALUES ('6','Georg Auerbacher','West Germany','1:18.14.6','5',3);
INSERT INTO  "climber" VALUES ('7','Arseneus Butscher','West Germany','1:21.35.6','4',5);
INSERT INTO  "climber" VALUES ('8','Charlie Freedman','United Kingdom','1:25.02.8','3',5);
INSERT INTO  "climber" VALUES ('9','L Currie','United Kingdom','1:25.40.6','2',7);
INSERT INTO  "climber" VALUES ('10','Mick Horsepole','United Kingdom','1:27.28.8','1',7);

