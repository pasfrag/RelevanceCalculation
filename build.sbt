name := "RelevanceCalculation"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "2.4.4", "org.apache.spark" %% "spark-sql" % "2.4.4", "org.apache.spark" %% "spark-mllib" % "2.4.4")

libraryDependencies += "com.github.master" % "spark-stemming_2.10" % "0.2.1"