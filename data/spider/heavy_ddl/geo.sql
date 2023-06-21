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
  highest_elevation text,
  lowest_point text,
  highest_point text,
  lowest_elevation text
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