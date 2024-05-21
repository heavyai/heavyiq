DROP DATABASE IF EXISTS flight_2;
CREATE DATABASE flight_2;
ALTER SESSION SET CURRENT_DATABASE = 'flight_2';

CREATE TABLE airlines (
	uid INTEGER , 
	Airline TEXT, 
	Abbreviation TEXT, 
	Country TEXT
);

CREATE TABLE airports (
	City TEXT, 
	AirportCode TEXT , 
	AirportName TEXT, 
	Country TEXT, 
	CountryAbbrev TEXT
);

CREATE TABLE flights (
	Airline INTEGER, 
	FlightNo INTEGER, 
	SourceAirport TEXT, 
	DestAirport TEXT);
