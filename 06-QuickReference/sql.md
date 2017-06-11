# Overview of the relational algebra and SQL

- Most of this page is just notes on [Data Manipulation at Scale: Systems and Algorithms](https://www.coursera.org/learn/data-manipulation).
- Databases: [SQLite](https://www.sqlite.org/), [MariaDB](http://mariadb.org/), [MySQL](https://www.mysql.com/), [PostgreSQL](https://www.postgresql.org/)
- Tutorials:
  [SQL tutorial](http://www.w3schools.com/sql/default.asp), 
  [SQL Reference](http://www.w3schools.com/sql/sql_quickref.asp),
  [Tutorial](http://zetcode.com/),
  [PostgreSQL Tutorial](http://www.postgresqltutorial.com/)
- This overview is obviously incomplete. 

# Relational Algebra and SQL

Relational Algebra is an algebra on tables and consists of the following operators

- Union §\cup§ , Intersection §\cap§ , Difference §-§ ,
- Selection §\sigma§ , Projection §\Pi§ ,
- (Natural) Join §\bowtie§ .

Note that the algebra is closed under the above operators, i.e. **the output is always a table**.

The relational algebra equipped with the following operators is called extended relational algebra.

- Duplicate Elimination §d§ , 
- Grouping and Aggregation §g§ ,
- Sorting §t§ .

## Union, Intersection and Difference

<table><tr><td>
R1 = &nbsp;
</td><td>

<table class="list">
<tr><td>A</td><td>B</td></tr>
<tr><td>a1</td><td>b1</td></tr>
<tr><td>a2</td><td>b1</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;&nbsp;
R2 = &nbsp;
</td><td>

<table class="list">
<tr><td>A</td><td>B</td></tr>
<tr><td>a1</td><td>b1</td></tr>
<tr><td>a3</td><td>b4</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;&nbsp;
R1 &cup; R2 = &nbsp;
</td><td>

<table class="list">
<tr><td>A</td><td>B</td></tr>
<tr><td>a1</td><td>b1</td></tr>
<tr><td>a2</td><td>b1</td></tr>
<tr><td>a1</td><td>b1</td></tr>
<tr><td>a3</td><td>b4</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;&nbsp;
R1 &cap; R2 = &nbsp;
</td><td>

<table class="list">
<tr><td>A</td><td>B</td></tr>
<tr><td>a1</td><td>b1</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;&nbsp;
R1 &minus; R2 = &nbsp;
</td><td>

<table class="list">
<tr><td>A</td><td>B</td></tr>
<tr><td>a2</td><td>b1</td></tr>
</table>

</td></tr></table>

- `SELECT * FROM R1 UNION ALL SELECT * FROM R2;`
  : the union R1 §\cup§ R2 
- `SELECT * FROM R1 INTERSECT SELECT * FROM R2;`
  : the intersection R1 §\cap§ R2 
- `SELECT * FROM R1  EXCEPT SELECT * FROM R2;`
  : the difference R1 §-§ R2 

Claim: R1 §\cap§ R2 = R1 §-§ (R1 §-§ R2). The RHS is obtained by
	
	SELECT * FROM R1 EXCEPT SELECT * FROM (SELECT * FROM R1 EXCEPT SELECT * FROM R2);

## Selection and Projection

<table><tr><td>
S = &nbsp;
</td><td>

<table class="list">
<tr><td>id</td><td>name</td><td>score</td></tr>
<tr><td>100</td><td>John</td><td>70</td></tr>
<tr><td>101</td><td>Smith</td><td>50</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;&nbsp;
&sigma;<sub>score&geq;60</sub>(S) = &nbsp;
</td><td>

<table class="list">
<tr><td>id</td><td>name</td><td>score</td></tr>
<tr><td>100</td><td>John</td><td>70</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;&nbsp;
&prod;<sub>name,score</sub>(S) = &nbsp;
</td><td>

<table class="list">
<tr><td>name</td><td>score</td></tr>
<tr><td>John</td><td>70</td></tr>
<tr><td>Smith</td><td>50</td></tr>
<tr><td>Fred</td><td>90</td></tr>
</table>

</td></tr></table>

### Selection

Pick rows satisfying a certain condition §c§. The result is written as §\sigma_c(R)§. Here §c§ is a map sending a row into a boolean value.

	SELECT * FROM S WHERE score >= 60;

We use `WHERE c` for the selection §\sigma_c§ , not `SELECT`.

- `=` is used for an equation. Not `==`.
- The condition `score >= 50 AND score <= 70` can be described simpler with the `BETWEEN` operator.

		SELECT * FROM S WHERE score BETWEEN 50 AND 70;

- For a condition §c§ we can use `LIKE` operator to find a pattern of a strings.

		SELECT name FROM S WHERE name LIKE '%h_';

  Here `%` means one or more characters and `_` means a single character.
- Instead of `LIKE` we can use the regular expression with `REGEXP` operator.

		SELECT name FROM S WHERE name REGEXP '^[aeiouAEIOU]';

- For a condition §c§ we can use `IN` operator to specify multiple values in a column.

		SELECT * FROM S WHERE score IN ('50','90');

- To obtain the first n rows in a table, use `LIMIT`.

		SELECT * FROM S LIMIT 50;

- Use `HAVING` instead of `WHERE` when using an aggregate function.

		SELECT * FROM S 

### Projection

Restrict a table to the specified columns, say A1, ... , An. The result is written as §\Pi§<sub>A1,...,An</sub>(R).

	SELECT name,score from S;

We use `SELECT name,score` for the projection §\Pi§<sub>name,score</sub>.

## Join and Cross Product

<table><tr><td>
S = &nbsp;
</td><td>

<table class="list">
<tr><td>id</td><td>name</td><td>score</td></tr>
<tr><td>100</td><td>John</td><td>70</td></tr>
<tr><td>101</td><td>Smith</td><td>50</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;
S2 = &nbsp;
</td><td>

<table class="list">
<tr><td>name</td><td>hight</td></tr>
<tr><td>Fred</td><td>165.0</td></tr>
<tr><td>John</td><td>181.0</td></tr>
<tr><td>Smith</td><td>172.0</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;
S × S2 = &nbsp;
</td><td>

<table class="list">
<tr><td>id</td><td>name</td><td>score</td><td>name</td><td>hight</td></tr>
<tr><td>100</td><td>John</td><td>70</td><td>Fred</td><td>165.0</td></tr>
<tr><td>100</td><td>John</td><td>70</td><td>John</td><td>181.0</td></tr>
<tr><td>100</td><td>John</td><td>70</td><td>Smith</td><td>172.0</td></tr>
<tr><td>101</td><td>Smith</td><td>50</td><td>Fred</td><td>165.0</td></tr>
<tr><td>101</td><td>Smith</td><td>50</td><td>John</td><td>181.0</td></tr>
<tr><td>101</td><td>Smith</td><td>50</td><td>Smith</td><td>172.0</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td><td>Fred</td><td>165.0</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td><td>John</td><td>181.0</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td><td>Smith</td><td>172.0</td></tr>
</table>

</td></tr></table>

The cross product R1§\times§R2 (a.k.a. the cross join) gives the all possible pair of a row of R1 and a row of R2. The cross proect is obtained by `SELECT * FROM S,S2;`. 

The **equi-join** R1 §\bowtie§<sub>A=B</sub> R2 is defined by §\sigma§<sub>A=B</sub>(R1§\times§R2). Therefore S §\bowtie§ <sub>S.name=S2.name</sub> S2 is the following table

<table class="list">
<tr><td>id</td><td>name</td><td>score</td><td>name</td><td>hight</td></tr>
<tr><td>100</td><td>John</td><td>70</td><td>John</td><td>181.0</td></tr>
<tr><td>101</td><td>Smith</td><td>50</td><td>Smith</td><td>172.0</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td><td>Fred</td><td>165.0</td></tr>
</table>

and we obtain it by `SELECT * FROM S,S2 WHERE S.name = S2.name;` by definition, or

	SELECT * FROM S JOIN S2 ON S.name = S2.name;

The **theta-join** R1 §\bowtie_\theta§ R2 is an easy extension of the equi-join and defined by §\sigma\_\theta§(R1§\times§R2). 

Outer joins: [left outer join](http://www.w3schools.com/sql/sql_join_left.asp), [right outer join](http://www.w3schools.com/sql/sql_join_right.asp), [full outer join](http://www.w3schools.com/sql/sql_join_full.asp)


## Duplicate Elimination and Sort

<table><tr><td>
S = &nbsp;
</td><td>

<table class="list">
<tr><td>id</td><td>name</td><td>score</td></tr>
<tr><td>100</td><td>John</td><td>70</td></tr>
<tr><td>101</td><td>Smith</td><td>50</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td></tr>
<tr><td>103</td><td>Anna</td><td>90</td></tr>
<tr><td>104</td><td>Maria</td><td>50</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;
d&prod;<sub>score</sub>(S) = &nbsp;
</td><td>

<table class="list">
<tr><td>score</td></tr>
<tr><td>70</td></tr>
<tr><td>50</td></tr>
<tr><td>90</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;
t<sub>score</sub>(S) = &nbsp;
</td><td>

<table class="list">
<tr><td>id</td><td>name</td><td>score</td></tr>
<tr><td>101</td><td>Smith</td><td>50</td></tr>
<tr><td>104</td><td>Maria</td><td>50</td></tr>
<tr><td>100</td><td>John</td><td>70</td></tr>
<tr><td>102</td><td>Fred</td><td>90</td></tr>
<tr><td>103</td><td>Anna</td><td>90</td></tr>
</table>

</td></tr></table>

Use `DISTINCT` to eliminate duplicates of rows. The second table is obtained by

	SELECT DISTINCT score FROM S;

Use `ORDER BY` to sort the rows by one or more columns. The third table is obtained by

	SELECT * FROM S ORDER BY score;

Note that the default order is the ascending order. Use the `DESC` keyword to sort the rows in a descending order. For example, `SELECT * FROM S ORDER BY score DESC, id DESC;` sorts the rows by the columns score and id in descending order .

## Groupby

<table><tr><td>
T = &nbsp;
</td><td>

<table class="list">
<tr><td>class</td><td>name</td><td>score</td></tr>
<tr><td>a</td><td>Fred</td><td>55</td></tr>
<tr><td>a</td><td>Maria</td><td>99</td></tr>
<tr><td>b</td><td>John</td><td>60</td></tr>
<tr><td>b</td><td>Julia</td><td>82</td></tr>
<tr><td>c</td><td>Britta</td><td>50</td></tr>
<tr><td>c</td><td>Smith</td><td>50</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;
T1 = &nbsp;
</td><td>

<table class="list">
<tr><td>class</td><td>average</td></tr>
<tr><td>a</td><td>77.0</td></tr>
<tr><td>b</td><td>71.0</td></tr>
<tr><td>c</td><td>50</td></tr>
</table>

</td><td>
&nbsp;&nbsp;&nbsp;
T2 = &nbsp;
</td><td>

<table class="list">
<tr><td>class</td></tr>
<tr><td>a</td></tr>
<tr><td>b</td></tr>
</table>

</td></tr></table>

The `GROUP BY` statement divides a table by a certain column. Applying it together with an aggregate function, we obtain a "summary" table.

Table T1 is the table of the averages of scores by class and obtained by

	SELECT class, AVG(score) AS average FROM T GROUP BY class;

Table T2 is the table of the classes in which there is a person whose score &gt; 60. It is obtained by

	SELECT class FROM T GROUP BY class HAVING MAX(score) > 60;

Note the `HAVING` keyword.

### Aggregate functions

- `AGV()` gives the mean value.
- `COUNT()` gives the number of rows. A typical usage is `COUNT(*)`.
- `MAX()`/`MIN()` gives the maximum/minimum value in the column.
- `SUM()` gives the sum of the values in the column.
- `POWER(,)` is used instead of `**`.
- `CEIL()`/`FLOOR()`
- `REPLACE(col,'a','A')` : replace substring with a new one

Some examples:

- `SELECT *, MAX(score) AS max FROM T GROUP BY class;` finds the top score by class.
- `SELECT class, COUNT(*) AS count FROM T GROUP BY class;` counts the number of rows in each class.


## Comparison between RA, SQL and R (with dplyr)

<table class="list">
<tr><td>RA</td><td>SQL</td><td>R/dplyr</td></tr>
<tr><td>union</td><td>UNION</td><td>rbind()</td></tr>
<tr><td>intersection</td><td>INTERSECT</td><td>intersect()</td></tr>
<tr><td>difference</td><td>EXCEPT</td><td>anti_join()</td></tr>
<tr><td>selection</td><td>WHERE</td><td>filter()</td></tr>
<tr><td>projection</td><td>SELECT</td><td>select()</td></tr>
<tr><td>join</td><td>JOIN</td><td>merge()</td></tr>
<tr><td>duplicate elimination</td><td>DISTINCT</td><td>distinct()</td></tr>
<tr><td>sorting</td><td>ORDER BY</td><td>arrange()</td></tr>
<tr><td>grouping</td><td>GROUP BY</td><td>group_by()</td></tr>
</table>

Notes for R

- The `sqldf` package allows us to work with SQL commands on R. [Manual](https://cran.r-project.org/web/packages/sqldf/sqldf.pdf), [Tutorial 1](http://www.burns-stat.com/translating-r-sql-basics/), [Tutorial 2](http://www.r-bloggers.com/make-r-speak-sql-with-sqldf/).
- The `plyr` package provides `join()` function. [Manual](http://www.inside-r.org/packages/cran/plyr/docs/join).
- `merge(df,dg,by=NULL)` gives a cross join.


# Modifying a Table

- Add a new row

		INSERT INTO Table VALUES (val1,val2,val3,...);
		INSERT INTO Table (col1,col2,col3,...) VALUES (val1,val2,val3,...);

- Modify an exsisting row

		UPDATE Table SET col1=val1,col2=val2,... WHERE coln=val;

- Delete a row

		DELETE FROM Table WHERE col=val;

# Managing Tables

- Create a table

		CREATE TABLE Table (A TEXT, B TEXT);
- Creates a virtual table.

		CREATE VIEW NewTable AS SELECT * FROM Table WHERE score>60;
  A virtual table always gives up-to-date data.
- Deletes a table

		DROP TABLE Table;

## Alter TABLE statement

- [Tutorial1](http://zetcode.com/databases/mysqltutorial/tables/#alter),
  [Tutorial2](http://www.w3schools.com/sql/sql_alter.asp).
- According to [SQLite FAQ](http://www.sqlite.org/faq.html#q11):
  (11) How do I add or delete columns from an existing table in SQLite.
  > SQLite has limited ALTER TABLE support that you can use to add a column to the end of a table or to change the name of a table. If you want to make more complex changes in the structure of a table, you will have to recreate the table. You can save existing data to a temporary table, drop the old table, create the new table, then copy the data back in from the temporary table.

## Datatype (schema)

- [SQLite](https://www.sqlite.org/datatype3.html): NULL, INTEGER, REAL, TEXT, BLOB.
- [MariaDB](https://mariadb.com/kb/en/mariadb/data-types/), [MySQL](https://dev.mysql.com/doc/refman/5.7/en/data-types.html), [Overview](http://www.w3resource.com/mysql/mysql-data-types.php).
  - Integer: `TINYINT`, `SMALLINT`, `MEDIUMINT`, `INT`, `BIGINT`
  - Text: `TINYTEXT`, `TEXT`, `MEDIUMTEXT`, `LONGTEXT`
  - Blob: `TINYBLOB`, `BLOB`, `MEDIUMBLOB`, `LONGBLOB`
  - Floating-Point: `FLOAT`, `DOUBLE`
  - Fixed-Point Types: `DECIMAL( , )`
  - Bit Value Types: `BIT(N)`
  - Date and Time Types: `DATETIME`, `DATE`, `TIMESTAMP`
  - CHAR and VARCHAR Types: `CHAR`, `VARCHAR`
  - BINARY and VARBINARY Types: `BINARY`, `VARBINARY`
  
## Show the information of a table

- SQLite: `pragma table_info(Table);`

		sqlite> pragma table_info(S);
		cid         name        type        notnull     dflt_value  pk        
		----------  ----------  ----------  ----------  ----------  ----------
		0           id          int         0                       0         
		1           name        text        0                       0         
		2           score       int         0                       0 

  `.schema` shows the commands which are used to create all tables.
- MySQL/MariaDB: `DESCRIBE Table`. [Manual](http://dev.mysql.com/doc/refman/5.1/en/describe.html). This is equivalent to `SHOW COLUMNS FROM Table;`. More details including character encoding can be obtained by `SHOW FULL COLUMNS FROM Table;`. [About UTF-8 on MySQL](http://50226.de/mysql-umlaute-und-utf-8.html) (German).

# Window functions

- Window functions are not available on SQLite and MySQL.
- [Tutorial 1](http://www.postgresqltutorial.com/postgresql-window-function/), [Tutorial 2](http://tapoueh.org/blog/2013/08/20-Window-Functions), [Tutorial 3](https://www.simple-talk.com/sql/t-sql-programming/window-functions-in-sql/), [window functions in dplyr](https://cran.r-project.org/web/packages/dplyr/vignettes/window-functions.html)

A fundamental syntax is:

	<window function> OVER (
		[PARTITION BY <expression list>]
		[ORDER BY <expression [ASC|DESC] list>]
		[ROWS|RANGE <window frame>]
	)


- The usage of `PARTITION BY` is the same as one of `GROUP BY`.
- When using `ORDER BY`, the window is going to be

		ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

  Assume we use `PARTITION BY` with `ORDER BY`. To make the range the whole of the corresponding group, use the following range

		ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING

  Such a frame could be needed when we use a "order sensitive" function such as  `first_value()`.
- `OVER ()` means all rows
- We can create an alias of a window by `WINDOW AS w (ORDER BY col)`.

		SELECT col, LAG(col,1) OVER w FROM generate_series(1,15,2) as T(col) WINDOW w AS (ORDER BY col);

### Window functions

- Every aggregate function (including a user defined aggregate function) can be used as a window function.
- The build-in [aggregate functions](https://www.postgresql.org/docs/9.2/static/functions-aggregate.html) / [window functions](https://www.postgresql.org/docs/9.2/static/functions-window.html) of PostgreSQL.
- `row_number()` gives the row index to each row.
- Some functions requires `ORDER BY <exp>` instead of an expression.
    - `rank()` / `dense_rank()` : give the rank according to `<exp>`
	- `percent_rank()` percenteil function
	- `cume_dist()` gives the cumulative distribution. 

In the following example, a table `q` contains only one column `x` with values 4, 2, 5, 1, 9, 4, 0, 6.

	select x,
		row_number() over (),
		rank() over w,
		dense_rank() over w,
		percent_rank() over w,
		cume_dist() over w
	from q window w as (order by x);

The above statement creates the following table. Note that the order of rows are changed because of the window `w`.

	 x | row_number | rank | dense_rank |   percent_rank    | cume_dist 
	---+------------+------+------------+-------------------+-----------
	 0 |          1 |    1 |          1 |                 0 |     0.125
	 1 |          2 |    2 |          2 | 0.142857142857143 |      0.25
	 2 |          3 |    3 |          3 | 0.285714285714286 |     0.375
	 4 |          4 |    4 |          4 | 0.428571428571429 |     0.625
	 4 |          5 |    4 |          4 | 0.428571428571429 |     0.625
	 5 |          6 |    6 |          5 | 0.714285714285714 |      0.75
	 6 |          7 |    7 |          6 | 0.857142857142857 |     0.875
	 9 |          8 |    8 |          7 |                 1 |         1
