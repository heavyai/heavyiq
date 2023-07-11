DROP DATABASE IF EXISTS epinions_1;
CREATE DATABASE epinions_1;
ALTER SESSION SET CURRENT_DATABASE = 'epinions_1';

CREATE TABLE item (
  i_id integer NOT NULL,
  title TEXT DEFAULT NULL
);

INSERT INTO item VALUES(0,'pear');
INSERT INTO item VALUES(1,'orange');
INSERT INTO item VALUES(2,'apple');
INSERT INTO item VALUES(3,'shampoo');
INSERT INTO item VALUES(4,'avocado');
INSERT INTO item VALUES(5,'comb');
INSERT INTO item VALUES(6,'blue hoodie');
INSERT INTO item VALUES(7,'cup');

CREATE TABLE review (
  a_id integer NOT NULL,
  u_id integer NOT NULL,
  i_id integer NOT NULL,
  rating integer DEFAULT NULL,
  "rank" integer DEFAULT NULL
);

INSERT INTO review VALUES(1,1,1,10,1);
INSERT INTO review VALUES(2,2,1,5,2);
INSERT INTO review VALUES(3,1,4,7,3);
INSERT INTO review VALUES(4,2,7,10,7);
INSERT INTO review VALUES(5,2,5,7,4);
INSERT INTO review VALUES(6,1,3,5,5);
INSERT INTO review VALUES(7,2,7,6,6);

CREATE TABLE useracct (
  u_id integer NOT NULL,
  name TEXT DEFAULT NULL
);

INSERT INTO useracct VALUES(1,'Helen');
INSERT INTO useracct VALUES(2,'Mark');
INSERT INTO useracct VALUES(3,'Terry');
INSERT INTO useracct VALUES(4,'Nancy');
INSERT INTO useracct VALUES(5,'Rosie');
INSERT INTO useracct VALUES(6,'Roxi');
INSERT INTO useracct VALUES(7,'Emily');

CREATE TABLE IF NOT EXISTS "trust" (
  source_u_id integer NOT NULL,
  target_u_id integer NOT NULL,
  trust integer NOT NULL
);
  
INSERT INTO trust VALUES(1,2,10);
INSERT INTO trust VALUES(1,3,6);
INSERT INTO trust VALUES(2,4,8);
INSERT INTO trust VALUES(3,6,10);
INSERT INTO trust VALUES(7,2,3);
INSERT INTO trust VALUES(7,5,2);
INSERT INTO trust VALUES(7,3,4);
INSERT INTO trust VALUES(6,2,1);
INSERT INTO trust VALUES(1,5,7);
