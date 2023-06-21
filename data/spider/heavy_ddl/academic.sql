DROP DATABASE IF EXISTS academic;
CREATE DATABASE academic;
ALTER SESSION SET CURRENT_DATABASE = 'academic';
CREATE TABLE "author" (
"aid" int,
"homepage" text,
"name" text,
"oid" int);
CREATE TABLE "conference" (
"cid" int,
"homepage" text,
"name" text);
CREATE TABLE "domain" (
"did" int,
"name" text);
CREATE TABLE "domain_author" (
"aid" int, 
"did" int);

CREATE TABLE "domain_conference" (
"cid" int,
"did" int);
CREATE TABLE "journal" (
"homepage" text,
"jid" int,
"name" text);
CREATE TABLE "domain_journal" (
"did" int,
"jid" int);
CREATE TABLE "keyword" (
"keyword" text,
"kid" int);
CREATE TABLE "domain_keyword" (
"did" int,
"kid" int);
CREATE TABLE "publication" (
"abstract" text,
"cid" text,
"citation_num" int,
"jid" int,
"pid" int,
"reference_num" int,
"title" text,
"year" int);
CREATE TABLE "domain_publication" (
"did" int,
"pid" int);

CREATE TABLE "organization" (
"continent" text,
"homepage" text,
"name" text,
"oid" int);

CREATE TABLE "publication_keyword" (
"pid" int,
"kid" int);
CREATE TABLE "writes" (
"aid" int,
"pid" int);
CREATE TABLE "cite" (
"cited" int,
"citing"  int);
