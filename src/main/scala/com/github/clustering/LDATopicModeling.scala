package com.github.clustering

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}

object LDATopicModeling {

  def main(args:Array[String])={

    val corpusPath = "src/main/resources/data/mini_newsgroups/*"
    val sparkSession = SparkSession.builder().appName(s"${this.getClass.getSimpleName}").master("local").getOrCreate()

    println("Reading data")

    val corpus = sparkSession.sparkContext.wholeTextFiles(corpusPath)
      .map(_._2.toLowerCase().split("\\s")).map(_.filter(_.length > 3).filter(_.forall(java.lang.Character.isLetter))).map(_.mkString(" "))

    val sqlContext = new SQLContext(sparkSession.sparkContext)
    import sqlContext.implicits._
    val df = corpus.toDF("corpus")

    println("Reading stopwords data")

    val stopWords = sparkSession.sparkContext.textFile("src/main/resources/stop_words").collect()

    val tokenizer = new RegexTokenizer().setPattern("\\W").setMinTokenLength(3)
      .setInputCol("corpus")
      .setOutputCol("rawTokens")

    val stopWordsRemover = new StopWordsRemover().setCaseSensitive(true)
      .setInputCol("rawTokens")
      .setOutputCol("tokens")
      .setStopWords(stopWords)

    val countVectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("features")

    println("Creating pipeline")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, countVectorizer))


    val model = pipeline.fit(df)
    val train = model.transform(df)

    val lda = new LDA().setOptimizer("online")
      .setK(20)
      .setMaxIter(3)

    println("Training LDA model")

    val ldaModel = lda.fit(train)

    ldaModel.write.overwrite().save("src/main/resources/saved_models/LDATopicModeling")

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    topicIndices.show(false)

  }

}
