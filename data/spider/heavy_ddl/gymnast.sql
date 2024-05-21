DROP DATABASE IF EXISTS gymnast;
CREATE DATABASE gymnast;
ALTER SESSION SET CURRENT_DATABASE = 'gymnast';

CREATE TABLE "gymnast" (
"Gymnast_ID" int,
"Floor_Exercise_Points" FLOAT,
"Pommel_Horse_Points" FLOAT,
"Rings_Points" FLOAT,
"Vault_Points" FLOAT,
"Parallel_Bars_Points" FLOAT,
"Horizontal_Bar_Points" FLOAT,
"Total_Points" FLOAT);

CREATE TABLE "people" (
"People_ID" int,
"Name" text,
"Age" FLOAT,
"Height" FLOAT,
"Hometown" text);


INSERT INTO  "people" VALUES (1,'Paul Hamm','24','1.71','Santo Domingo');
INSERT INTO  "people" VALUES (2,'Lorraine Súarez Carmona','21','1.75','Bonao');
INSERT INTO  "people" VALUES (3,'Ashley Pérez Cabrera','19','1.70','Miami');
INSERT INTO  "people" VALUES (4,'Elizabeth Quiñónez Aroyo','20','1.71','Santo Domingo');
INSERT INTO  "people" VALUES (5,'Eve Tatiana Cruz Oviedo','19','1.72','Santo Domingo');
INSERT INTO  "people" VALUES (6,'Nadia Caba Rodríguez','22','1.79','Santo Domingo');
INSERT INTO  "people" VALUES (7,'Clary Sermina Delgado Cid','21','1.75','Santiago de los Caballeros');
INSERT INTO  "people" VALUES (8,'Marina Castro Medina','20','1.76','Santo Domingo');
INSERT INTO  "people" VALUES (9,'Rosa Clarissa Ortíz Melo','23','1.81','La Romana');
INSERT INTO  "people" VALUES (10,'Endis de los Santos Álvarez','24','1.72','Los Alcarrizos');


INSERT INTO  "gymnast" VALUES ('1','9.725','9.737','9.512','9.575','9.762','9.750','58.061');
INSERT INTO  "gymnast" VALUES ('2','9.700','9.625','9.625','9.650','9.587','9.737','57.924');
INSERT INTO  "gymnast" VALUES ('4','8.987','9.750','9.750','9.650','9.787','9.725','57.649');
INSERT INTO  "gymnast" VALUES ('6','9.762','9.325','9.475','9.762','9.562','9.550','57.436');
INSERT INTO  "gymnast" VALUES ('7','9.687','9.675','9.300','9.537','9.725','9.500','57.424');
INSERT INTO  "gymnast" VALUES ('8','9.650','9.712','9.487','9.637','9.500','9.412','57.398');
INSERT INTO  "gymnast" VALUES ('10','9.412','9.525','9.712','9.550','9.625','9.550','57.374');

