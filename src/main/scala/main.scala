import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, MinHashLSH, MinHashLSHModel, StopWordsRemover, Tokenizer, VectorAssembler, Word2Vec}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.{Column, DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, udf, collect_set, concat_ws}

object main {

    def main(args: Array[String]): Unit = {

        // Starting the session
        val spark: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        import spark.implicits._
        val sc: SparkContext = spark.sparkContext
        val sqlContext: SQLContext = spark.sqlContext
        LogManager.getRootLogger.setLevel(Level.ERROR)

        // Reading the data
        val trainDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/train.csv")
        val descDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/product_descriptions.csv")
        val attrDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/attributes.csv")

        // General used variables
        val tokenizer = new Tokenizer().setInputCol("product_title").setOutputCol("title_tokenized")
        val remover = new StopWordsRemover()
        val stemmer = new Stemmer().setLanguage("English")
        val hashingTF = new HashingTF()
        val idf = new IDF()
        val word2Vec = new Word2Vec()
        val mh = new MinHashLSH()

        //      -----------------------------Training Data Preprocess-----------------------------------------------
        // Tokenizing the training data
        val semiTokenizedTrainDF = tokenizer.transform(trainDF)
        val tokenizedTrainDF = tokenizer.setInputCol("search_term").setOutputCol("search_tokenized")
          .transform(semiTokenizedTrainDF)

        //Removing stopwords from training data
        val semiFilteredTrainDF = remover.setInputCol("title_tokenized").setOutputCol("title_filtered")
          .transform(tokenizedTrainDF)
        val filteredTrainDF = remover.setInputCol("search_tokenized").setOutputCol("search_filtered")
          .transform(semiFilteredTrainDF)


        // Stemming Training data
        val semiStemmedTrainDF = stemmer.setInputCol("title_filtered").setOutputCol("title_stemmed")
          .transform(filteredTrainDF)
        val stemmedTrainDF = stemmer.setInputCol("search_filtered").setOutputCol("search_stemmed")
          .transform(semiStemmedTrainDF)

        // TF-IDF training data
        val semiFeaturesTrainDF = hashingTF.setInputCol("title_stemmed").setOutputCol("title_features").setNumFeatures(10000)
          .transform(stemmedTrainDF)
        val featuresTrainDF = hashingTF.setInputCol("search_stemmed").setOutputCol("search_features").setNumFeatures(10000)
          .transform(semiFeaturesTrainDF)
        val semiRescaledTRAINDF = idf.setInputCol("title_features").setOutputCol("title_tfidf").fit(featuresTrainDF)
          .transform(featuresTrainDF)
        val rescaledTrainDF = idf.setInputCol("search_features").setOutputCol("search_tfidf").fit(semiRescaledTRAINDF)
          .transform(semiRescaledTRAINDF)

        // Word2Vec to training data
        val semiVecTrainDF = word2Vec.setInputCol("title_stemmed").setOutputCol("title_w2v").fit(rescaledTrainDF)
          .transform(rescaledTrainDF)
        val vecTrainDF = word2Vec.setInputCol("search_stemmed").setOutputCol("search_w2v").fit(semiVecTrainDF)
          .transform(semiVecTrainDF)
        vecTrainDF.printSchema()
        vecTrainDF.take(10).foreach(x => println(x))

        //      -----------------------------Description Data Preprocess-----------------------------------------------
        // Tokenizing the description data
        val tokenizedDescriptionDF = tokenizer.setInputCol("product_description").setOutputCol("description_tokenized")
          .transform(descDF)

        // Removing stopwords from description data
        val filteredDescriptionDF = remover.setInputCol("description_tokenized").setOutputCol("description_filtered")
          .transform(tokenizedDescriptionDF)

        // Stemming description data
        val stemmedDescriptionDF = stemmer.setInputCol("description_filtered").setOutputCol("description_stemmed")
          .transform(filteredDescriptionDF)

        // TF-IDF description data
        val featuresDescriptionDF = hashingTF.setInputCol("description_stemmed").setOutputCol("description_features").setNumFeatures(10000)
          .transform(stemmedDescriptionDF)
        val rescaledDescriptionDF = idf.setInputCol("description_features").setOutputCol("description_tfidf").fit(featuresDescriptionDF)
          .transform(featuresDescriptionDF)

        // Word2Vec to description data
        val vecDescDF = word2Vec.setInputCol("description_stemmed").setOutputCol("description_w2v")
          .fit(rescaledDescriptionDF).transform(rescaledDescriptionDF)
        vecDescDF.printSchema()
        vecDescDF.take(10).foreach(x => println(x))


        //-----------------------------Attribute Data Preprocess-----------------------------------------------

        val result1 = attrDF.groupBy("product_uid").agg(collect_set("name"))
        val result2 = attrDF.groupBy("product_uid").agg(collect_set("value"))
        val result3 = result1.join(result2, result2("product_uid") === result1("product_uid"), "inner")
          .select(result1("product_uid"), result1("collect_set(name)"), result2("collect_set(value)"))
        val result4 = result3
          .withColumn("name", concat_ws(",", result3("collect_set(name)")))
          .withColumn("value", concat_ws(",", result3("collect_set(value)")))
          .orderBy("product_uid").select("product_uid", "name", "value")

        // Tokenizing the attribute data
        val semiTokenizedAttributeDF = tokenizer.setInputCol("name").setOutputCol("name_tokenized")
          .transform(result4)
        val tokenizedAttributeDF = tokenizer.setInputCol("value").setOutputCol("value_tokenized")
          .transform(semiTokenizedAttributeDF)

        // Removing stopwords from attribute data
        val semiFilteredAttributeDF = remover.setInputCol("name_tokenized").setOutputCol("name_filtered")
          .transform(tokenizedAttributeDF)
        val filteredAttributeDF = remover.setInputCol("value_tokenized").setOutputCol("value_filtered")
          .transform(semiFilteredAttributeDF)

        // Stemming description data
        val semiStemmedAttributeDF = stemmer.setInputCol("name_filtered").setOutputCol("name_stemmed").transform(filteredAttributeDF)
        val stemmedAttributeDF = stemmer.setInputCol("value_filtered").setOutputCol("value_stemmed").transform(semiStemmedAttributeDF)

        // TF-IDF attribute data
        val semiFeaturesAttributeDF = hashingTF.setInputCol("name_stemmed").setOutputCol("name_features").setNumFeatures(10000)
          .transform(stemmedAttributeDF)
        val featuresAttributeDF = hashingTF.setInputCol("value_stemmed").setOutputCol("value_features").setNumFeatures(10000)
          .transform(semiFeaturesAttributeDF)
        val semiRescaledAttributeDF = idf.setInputCol("name_features").setOutputCol("name_tfidf").fit(featuresAttributeDF)
          .transform(featuresAttributeDF)
        val rescaledAttributeDF = idf.setInputCol("value_features").setOutputCol("value_tfidf").fit(semiRescaledAttributeDF)
          .transform(semiRescaledAttributeDF)

        // Word2Vec to attribute data
        val semiVecAttrDF = word2Vec.setInputCol("name_stemmed").setOutputCol("name_w2v").fit(rescaledAttributeDF)
        .transform(rescaledAttributeDF)
        val vecAttrDF = word2Vec.setInputCol("value_stemmed").setOutputCol("value_w2v").fit(semiVecAttrDF)
          .transform(semiVecAttrDF)
        vecAttrDF.printSchema()
        vecAttrDF.take(10).foreach(x => println(x))


        //-----------------------------------------Test Area 51-----------------------------------------

        val udfToDouble = udf((s: String) => s.toDouble)
        val assembler = (new VectorAssembler().setInputCols(Array(
            "title_w2v","search_w2v", "title_tfidf","search_tfidf")).setOutputCol("features"))
        // Transform the DataFrame
        val output = assembler.transform(vecTrainDF).select(udfToDouble(col("relevance")).alias("label"),$"features")
        println(output.count())

        // Splitting the data by create an array of the training and test data
        val Array(training, test) = output.select("label","features").
          randomSplit(Array(0.6, 0.4), seed = 1L)//, seed = 12345
        println(training.count())
        println(test.count())
        val lir = new LinearRegression()
          .setFeaturesCol("features")
          .setLabelCol("label")
        //.setMaxIter(100)

        // You can then treat this object as the model and use fit on it.
        val lirModel = lir.fit(training)
        // Print the weights and intercept for linear regression.

//        println(s"Weights: ${lirModel.coefficients} Intercept: ${lirModel.intercept}")
//        println(s"Coefficients: ${lirModel.coefficients}")
//        println(s"Intercept: ${lirModel.intercept}")

        // Summarize the model over the training set and print out some metrics
//        val trainingSummary = lirModel.summary
//        println(s"numIterations: ${trainingSummary.totalIterations}")
//        println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
//        trainingSummary.residuals.show()
//        println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
//        println(s"r2: ${trainingSummary.r2}")

        //Creation of the evaluator
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("mse")

        //evaluator.printSchema()
        //evaluator.take(10).foreach(x=>print(x))

        val result = lirModel.transform(test)
        println(result.count())
        val MSElr = evaluator.evaluate(result)
        println("Linear Regression MSE = " + MSElr)
    }


//  ----------------------------------ML PART-------------------------------------------------------

    //TODO In each one search term and title. Then combined with description and attributes (not so sure)
    //TODO 5)LR with TFIDF
    //TODO 6)LR with WORD2VEC
    //TODO 7)LR with both above methods
    //TODO 8)Add cosine similarities???
    //TODO 9)DT with TFIDF
    //TODO 10)DT with WORD2VEC
    //TODO 11)DT with both above methods
    //TODO 12)Add cosine similarities???
    //TODO 13)RF with TFIDF
    //TODO 14)RF with WORD2VEC
    //TODO 15)RF with both above methods
    //TODO 16)Add cosine similarities???
    //TODO 17)LR with TFIDF
    //TODO 18)LR with WORD2VEC
    //TODO 19)LR with both above methods
    //TODO 20)Add cosine similarities???
    //TODO 21)Unsupervised part!!!!
}
