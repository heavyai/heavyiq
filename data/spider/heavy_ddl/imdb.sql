DROP DATABASE IF EXISTS imdb;
CREATE DATABASE imdb;
ALTER SESSION SET CURRENT_DATABASE = 'imdb';
CREATE TABLE "actor" (
"aid" int,
"gender" text,
"name" text,
"nationality" text,
"birth_city" text,
"birth_year" int);


CREATE TABLE "copyright" (
"id" int,
"msid" int,
"cid" int);
CREATE TABLE "cast" (
"id" int,
"msid" int,
"aid" int,
"role" int);

CREATE TABLE "genre" (
"gid" int,
"genre" text);

CREATE TABLE "classification" (
"id" int,
"msid" int,
"gid" int);

CREATE TABLE "company" (
"id" int,
"name" text,
"country_code" text);


CREATE TABLE "director" (
"did" int,
"gender" text,
"name" text,
"nationality" text,
"birth_city" text,
"birth_year" int);

CREATE TABLE "producer" (
"pid" int,
"gender" text,
"name" text,
"nationality" text,
"birth_city" text,
"birth_year" int);

CREATE TABLE "directed_by" (
"id" int,
"msid" int,
"did" int);

CREATE TABLE "keyword" (
"id" int,
"keyword" text);

CREATE TABLE "made_by" (
"id" int,
"msid" int,
"pid" int);

CREATE TABLE "movie" (
"mid" int,
"title" text,
"release_year" int,
"title_aka" text,
"budget" text);
CREATE TABLE "tags" (
"id" int,
"msid" int,
"kid" int);
CREATE TABLE "tv_series" (
"sid" int,
"title" text,
"release_year" int,
"num_of_seasons" int,
"num_of_episodes" int,
"title_aka" text,
"budget" text);
CREATE TABLE "writer" (
"wid" int,
"gender" text,
"name" int,
"nationality" int,
"num_of_episodes" int,
"birth_city" text,
"birth_year" int);
CREATE TABLE "written_by" (
"id" int,
"msid" int,
"wid" int);
