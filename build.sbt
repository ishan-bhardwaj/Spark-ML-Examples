name := "Spark-ML-Examples"

version := "1.0"

scalaVersion := "2.11.12"

val sparkVersion = "2.2.0"

resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion

libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion

libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
        