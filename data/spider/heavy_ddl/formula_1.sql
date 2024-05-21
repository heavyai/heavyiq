DROP DATABASE IF EXISTS formula_1;
CREATE DATABASE formula_1;
ALTER SESSION SET CURRENT_DATABASE = 'formula_1';

CREATE TABLE IF NOT EXISTS "circuits" (
"circuitId" INTEGER ,
  "circuitRef" TEXT,
  "name" TEXT,
  "location" TEXT,
  "country" TEXT,
  "lat" FLOAT,
  "lng" FLOAT,
  "alt" INTEGER,
  "url" TEXT
);
CREATE TABLE IF NOT EXISTS "races" (
"raceId" INTEGER ,
  "year" INTEGER,
  "round" INTEGER,
  "circuitId" INTEGER,
  "name" TEXT,
  "date" TEXT,
  "time" TEXT,
  "url" TEXT);

CREATE TABLE IF NOT EXISTS "drivers" (
"driverId" INTEGER ,
  "driverRef" TEXT,
  "number" INTEGER,
  "code" TEXT,
  "forename" TEXT,
  "surname" TEXT,
  "dob" TEXT,
  "nationality" TEXT,
  "url" TEXT
);
CREATE TABLE IF NOT EXISTS "status" (
"statusId" INTEGER ,
  "status" TEXT
);
CREATE TABLE IF NOT EXISTS "seasons" (
"year" INTEGER ,
  "url" TEXT
);
CREATE TABLE IF NOT EXISTS "constructors" (
	"constructorId" INTEGER ,
  "constructorRef" TEXT,
  "name" TEXT,
  "nationality" TEXT,
  "url" TEXT
);
CREATE TABLE IF NOT EXISTS "constructorStandings" (
	"constructorStandingsId" INTEGER ,
  "raceId" INTEGER,
  "constructorId" INTEGER,
  "points" FLOAT,
  "position" INTEGER,
  "positionText" TEXT,
  "wins" INTEGER);
CREATE TABLE IF NOT EXISTS "results" (
"resultId" INTEGER ,
  "raceId" INTEGER,
  "driverId" INTEGER,
  "constructorId" INTEGER,
  "number" INTEGER,
  "grid" INTEGER,
  "position" INTEGER,
  "positionText" TEXT,
  "positionOrder" INTEGER,
  "points" FLOAT,
  "laps" INTEGER,
  "time" TEXT,
  "milliseconds" INTEGER,
  "fastestLap" INTEGER,
  "rank" INTEGER,
  "fastestLapTime" TEXT,
  "fastestLapSpeed" TEXT,
  "statusId" INTEGER);
CREATE TABLE IF NOT EXISTS "driverStandings" (
"driverStandingsId" INTEGER ,
  "raceId" INTEGER,
  "driverId" INTEGER,
  "points" FLOAT,
  "position" INTEGER,
  "positionText" TEXT,
  "wins" INTEGER);
CREATE TABLE IF NOT EXISTS "constructorResults" (
"constructorResultsId" INTEGER ,
  "raceId" INTEGER,
  "constructorId" INTEGER,
  "points" FLOAT,
  "status" FLOAT);
CREATE TABLE IF NOT EXISTS "qualifying" (
"qualifyId" INTEGER ,
  "raceId" INTEGER,
  "driverId" INTEGER,
  "constructorId" INTEGER,
  "number" INTEGER,
  "position" INTEGER,
  "q1" TEXT,
  "q2" TEXT,
  "q3" TEXT);
CREATE TABLE IF NOT EXISTS "pitStops" (
"raceId" INTEGER,
  "driverId" INTEGER,
  "stop" INTEGER,
  "lap" INTEGER,
  "time" TEXT,
  "duration" TEXT,
  "milliseconds" INTEGER);
CREATE TABLE IF NOT EXISTS "lapTimes" (
"raceId" INTEGER,
  "driverId" INTEGER,
  "lap" INTEGER,
  "position" INTEGER,
  "time" TEXT,
  "milliseconds" INTEGER);
