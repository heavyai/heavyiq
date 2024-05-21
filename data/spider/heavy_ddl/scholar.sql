DROP DATABASE IF EXISTS scholar;
CREATE DATABASE scholar;
ALTER SESSION SET CURRENT_DATABASE = 'scholar';



CREATE TABLE venue (
  venueId integer NOT NULL
,  venueName TEXT DEFAULT NULL
);



CREATE TABLE author (
  authorId integer NOT NULL
,  authorName TEXT DEFAULT NULL
);


CREATE TABLE dataset (
  datasetId integer NOT NULL
,  datasetName TEXT DEFAULT NULL
);


CREATE TABLE journal (
  journalId integer NOT NULL
,  journalName TEXT DEFAULT NULL
);

CREATE TABLE keyphrase (
  keyphraseId integer NOT NULL
,  keyphraseName TEXT DEFAULT NULL
);


CREATE TABLE paper (
  paperId integer NOT NULL
,  title TEXT DEFAULT NULL
,  venueId integer DEFAULT NULL
,  "year" integer DEFAULT NULL
,  numCiting integer DEFAULT NULL
,  numCitedBy integer DEFAULT NULL
,  journalId integer DEFAULT NULL
);



CREATE TABLE cite (
  citingPaperId integer NOT NULL
,  citedPaperId integer NOT NULL
);


CREATE TABLE paperDataset (
  paperId integer DEFAULT NULL
,  datasetId integer DEFAULT NULL
);



CREATE TABLE paperKeyphrase (
  paperId integer DEFAULT NULL
,  keyphraseId integer DEFAULT NULL
);


CREATE TABLE writes (
  paperId integer DEFAULT NULL
,  authorId integer DEFAULT NULL
);

