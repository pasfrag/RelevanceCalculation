import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.sql.{Column, DataFrame, SQLContext, SparkSession}

object main {

    def main(args: Array[String]): Unit = {

        // Starting the session
        val spark: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        val sc: SparkContext = spark.sparkContext
        val sqlContext: SQLContext = spark.sqlContext
        LogManager.getRootLogger.setLevel(Level.ERROR)

        // Reading the data
        val trainDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/train.csv")
        val descDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/product_descriptions.csv")
        val attrDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/attributes.csv")


//      -----------------------------Training Data Preprocess-----------------------------------------------
        // Tokenizing the training data
        val tokenizer = new Tokenizer().setInputCol("product_title").setOutputCol("title_tokenized")
        val semiTokenizedTrainDF = tokenizer.transform(trainDF)
        tokenizer.setInputCol("search_term").setOutputCol("search_tokenized")
        val tokenizedTrainDF = tokenizer.transform(semiTokenizedTrainDF)
        val columnsToRemove = Seq("id", "product_title", "search_term")
        val removedTrainDF = tokenizedTrainDF.select(tokenizedTrainDF.columns
          .filter(colName => !columnsToRemove.contains(colName))
          .map(colName => new Column(colName)): _*
        )

        //Removing stopwords from training data
        val remover = new StopWordsRemover()
        remover.setInputCol("title_tokenized").setOutputCol("title_filtered")
        val semiFilteredTrainDF = remover.transform(removedTrainDF)
        remover.setInputCol("search_tokenized").setOutputCol("search_filtered")
        val filteredTrainDF = remover.transform(semiFilteredTrainDF)
        val finalTrainDF = filteredTrainDF.select(filteredTrainDF.columns
          .filter(colName => !Seq("title_tokenized", "search_tokenized").contains(colName))
          .map(colName => new Column(colName)): _*
        )

        //TODO 2)Lemmatization to training data

        // TF-ing training data
        val hashingTF = new HashingTF()
        hashingTF.setInputCol("title_filtered").setOutputCol("title_features").setNumFeatures(10000)
        val semiFeaturesTrainDF = hashingTF.transform(finalTrainDF)
        hashingTF.setInputCol("search_filtered").setOutputCol("search_features").setNumFeatures(10000)
        val featuresTrainDF = hashingTF.transform(semiFeaturesTrainDF)

        // TF-IDF training data
        val idf = new IDF()
        idf.setInputCol("title_features").setOutputCol("title")
        val semiTrainIDFModel = idf.fit(featuresTrainDF)
        val semiRescaledTRAINDF = semiTrainIDFModel.transform(featuresTrainDF)
        idf.setInputCol("search_features").setOutputCol("search")
        val trainIDFModel = idf.fit(semiRescaledTRAINDF)
        val rescaledTrainDF = trainIDFModel.transform(semiRescaledTRAINDF)
        val trainTFIDF = rescaledTrainDF.select(rescaledTrainDF.columns
          .filter(colName => !Seq("title_filtered", "title_features", "search_filtered", "search_features").contains(colName))
          .map(colName => new Column(colName)): _*
        )
//        trainTFIDF.printSchema()
//        trainTFIDF.take(10).foreach(x => print(x))

        // Word2Vec to training data
        val word2Vec = new Word2Vec()
        word2Vec.setInputCol("title_filtered").setOutputCol("title").setMinCount(0)//.setVectorSize(50)
        val semiTrainVecModel = word2Vec.fit(finalTrainDF)
        val semiVecTrainDF = semiTrainVecModel.transform(finalTrainDF)
        word2Vec.setInputCol("search_filtered").setOutputCol("search").setMinCount(0)//.setVectorSize(50)
        val trainVecModel = word2Vec.fit(semiVecTrainDF)
        val vecTrainDF = trainVecModel.transform(semiVecTrainDF)
        val trainW2VDF = vecTrainDF.select(vecTrainDF.columns
          .filter(colName => !Seq("title_filtered", "search_filtered").contains(colName))
          .map(colName => new Column(colName)): _*
        )
//        trainW2VDF.printSchema()
//        trainW2VDF.take(10).foreach(x => print(x))


//      -----------------------------Description Data Preprocess-----------------------------------------------
        // Tokenizing the description data
        tokenizer.setInputCol("product_description").setOutputCol("description_tokenized")
        val tokenizedDescriptionDF = tokenizer.transform(descDF)
        val removedDescriptionDF = tokenizedDescriptionDF.select(tokenizedDescriptionDF.columns
          .filter(colName => colName != "product_description")
          .map(colName => new Column(colName)): _*
        )

        // Removing stopwords from description data
        remover.setInputCol("description_tokenized").setOutputCol("description_filtered")
        val filteredDescriptionDF = remover.transform(removedDescriptionDF)
        val finalDescriptionDF = filteredDescriptionDF.select(filteredDescriptionDF.columns
            .filter(colName => colName != "description_tokenized")
            .map(colName => new Column(colName)): _*
        )

        //TODO 3)Lemmatization to description data

        // TF-ing description data
        hashingTF.setInputCol("description_filtered").setOutputCol("description_features").setNumFeatures(10000)
        val featuresDescriptionDF = hashingTF.transform(finalDescriptionDF)

        // TF-IDF description data
        idf.setInputCol("description_features").setOutputCol("description")
        val descriptionIDFModel = idf.fit(featuresDescriptionDF)
        val rescaledDescriptionDF = descriptionIDFModel.transform(featuresDescriptionDF)
        val descriptionTFIDF = rescaledDescriptionDF.select(rescaledDescriptionDF.columns
          .filter(colName => !Seq("description_filtered", "description_features").contains(colName))
          .map(colName => new Column(colName)): _*
        )
//        descriptionTFIDF.printSchema()
//        descriptionTFIDF.take(10).foreach(x => print(x))

        // Word2Vec to description data
        word2Vec.setInputCol("description_filtered").setOutputCol("description").setMinCount(0).setVectorSize(50)
        val descVecModel = word2Vec.fit(finalDescriptionDF)
        val vecDescDF = descVecModel.transform(finalDescriptionDF)
        val descriptionW2VDF = vecDescDF.select(vecDescDF.columns
          .filter(colName => !Seq("description_filtered").contains(colName))
          .map(colName => new Column(colName)): _*
        )
        descriptionW2VDF.printSchema()
        descriptionW2VDF.take(10).foreach(x => print(x))


//      -----------------------------Attribute Data Preprocess-----------------------------------------------
        // Tokenizing the attribute data
        tokenizer.setInputCol("name").setOutputCol("name_tokenized")
        val semiTokenizedAttributeDF = tokenizer.transform(attrDF)
        tokenizer.setInputCol("value").setOutputCol("value_tokenized")
        val tokenizedAttributeDF = tokenizer.transform(semiTokenizedAttributeDF)
        val columnsToRemove1 = Seq("name", "value")
        val removedAttributeDF = tokenizedAttributeDF.select(tokenizedAttributeDF.columns
          .filter(colName => !columnsToRemove1.contains(colName))
          .map(colName => new Column(colName)): _*
        )

        // Removing stopwords from attribute data
        remover.setInputCol("name_tokenized").setOutputCol("name_filtered")
        val semiFilteredAttributeDF = remover.transform(removedAttributeDF)
        remover.setInputCol("value_tokenized").setOutputCol("value_filtered")
        val filteredAttributeDF = remover.transform(semiFilteredAttributeDF)
        val finalAttributeDF = filteredAttributeDF.select(filteredAttributeDF.columns
          .filter(colName => !Seq("name_tokenized", "value_tokenized").contains(colName))
          .map(colName => new Column(colName)): _*
        )

        //TODO 4)Lemmatization to attribute data

        // TF-ing attribute data
        hashingTF.setInputCol("name_filtered").setOutputCol("name_features").setNumFeatures(10000)
        val semiFeaturesAttributeDF = hashingTF.transform(finalAttributeDF)
        hashingTF.setInputCol("value_filtered").setOutputCol("value_features").setNumFeatures(10000)
        val featuresAttributeDF = hashingTF.transform(semiFeaturesAttributeDF)

        //TODO 1)Make this shit work!!!

        // TF-IDF attribute data
//        idf.setInputCol("name_features").setOutputCol("name")
//        val semiAttributeIDFModel = idf.fit(featuresAttributeDF)
//        val semiRescaledAttributeIDF = semiAttributeIDFModel.transform(featuresAttributeDF)
//        idf.setInputCol("value_features").setOutputCol("value")
//        val attributeIDFModel = idf.fit(semiRescaledAttributeIDF)
//        val rescaledAttributeDF = attributeIDFModel.transform(semiRescaledAttributeIDF)

//        idf.setInputCol("name_features").setOutputCol("name")
//        val semiAttributeIDFModel = idf.fit(featuresAttributeDF)
//        val semiRescaledAttributeDF = semiAttributeIDFModel.transform(featuresAttributeDF)
//        idf.setInputCol("value_features").setOutputCol("value")
//        val attributeIDFModel = idf.fit(semiRescaledAttributeDF)
//        val rescaledAttributeDF = attributeIDFModel.transform(semiRescaledAttributeDF)
//        val attributeTFIDF = rescaledAttributeDF.select(rescaledAttributeDF.columns
//          .filter(colName => !Seq("name_filtered", "name_features", "value_filtered", "value_features").contains(colName))
//          .map(colName => new Column(colName)): _*
//        )
//        attributeTFIDF.printSchema()
//        attributeTFIDF.take(10).foreach(x => print(x))

//        trainDF.take(20).foreach(x => print(x))
//        descDF.take(20).foreach(x => print(x))
//        attrDF.take(20).foreach(x => print(x))
//
//        not useful
//        trainDF.createOrReplaceTempView("train_desc")
//        descDF.createOrReplaceTempView("desc_desc")
//        attrDF.createOrReplaceTempView("attr_desc")
//
//        val trainWithDescAndAttrDF:DataFrame = trainDF.join(descDF, descDF("product_uid") === trainDF("product_uid"))
//            .join(attrDF, attrDF("product_uid") === trainDF("product_uid"))
//
//        sqlContext.sql(
//            """SELECT a.id, a.product_uid, a.product_title, a.search_term, a.relevance, b.product_description, c.name, c.value FROM train_desc as a INNER JOIN desc_desc as b on a.product_uid = b.product_uid INNER JOIN attr_desc as c on a.product_uid = c.product_uid GROUP BY""")
//
//        trainWithDescAndAttrDF.take(50).foreach(x => print(x))
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
