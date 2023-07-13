DROP DATABASE IF EXISTS geo;
CREATE DATABASE geo;
ALTER SESSION SET CURRENT_DATABASE = 'geo';

CREATE TABLE state (
   state_name text,
   population integer DEFAULT NULL,
   area double DEFAULT NULL,
   country_name TEXT NOT NULL DEFAULT '',
   capital text,
   density double DEFAULT NULL
);

CREATE TABLE city (
  city_name text,
  population integer DEFAULT NULL,
  country_name TEXT NOT NULL DEFAULT '',
  state_name text
);
CREATE TABLE border_info (
  state_name text,
   border text
);
CREATE TABLE highlow (
  state_name text,
  highest_elevation integer,
  lowest_point text,
  highest_point text,
  lowest_elevation integer
);
CREATE TABLE lake (
  lake_name text,
  area double DEFAULT NULL,
  country_name TEXT NOT NULL DEFAULT '',
  state_name text
);
CREATE TABLE mountain (
  mountain_name text,
  mountain_altitude integer DEFAULT NULL,
  country_name TEXT NOT NULL DEFAULT '',
  state_name text
);
CREATE TABLE river (
  river_name text,
  length integer DEFAULT NULL,
  country_name TEXT NOT NULL DEFAULT '',
  traverse text
);

INSERT INTO highlow VALUES ('colorado', 14440, 'Yuma County', 'Mount Elbert', 3317);
INSERT INTO highlow VALUES ('alabama', 2407, 'Gulf of Mexico', 'Cheaha Mountain', 60);
INSERT INTO highlow VALUES ('new york', 5344, 'Atlantic Ocean', 'Mount Marcy', 0);
INSERT INTO state VALUES ('texas', 28296099, 65.05964231171514, 'united states', 'austin', 435.015);
INSERT INTO state VALUES ('new york', 19570261, 54.556, 'united states', 'new york', 358.9);
INSERT INTO state VALUES ('massachusetts', 6692824, 10.555, 'united states', 'boston', 633.9);
INSERT INTO city VALUES ('new york city', 8175133, 'united states', 'new york');