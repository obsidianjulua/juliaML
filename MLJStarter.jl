# MLJ Starting Models for Behavioral Intent Classification
# Five diverse models with varying strengths for constructivist learning

module MLJStarter

using MLJ
using MLJBase
using DataFrames
using Random

# Import specific model types
import MLJDecisionTreeInterface: DecisionTreeClassifier
import MLJScikitLearnInterface: RandomForestClassifier, SVC, MultinomialNB
import MLJXGBoostInterface: XGBoostClassifier

export create_starting_models, evaluate_model_ensemble, select_best_model

"""
Five Starting Models for Behavioral Intent Classification
========================================================

1. RandomForest - Robust ensemble, handles mixed features well
2. DecisionTree - Fast, interpretable, good for rule extraction  
3. XGBoost - High performance gradient boosting
4. MultinomialNB - Excellent for text/categorical features
5. SVC - Strong with high-dimensional behavioral data
"""

function create_starting_models()
    models = Dict{String, Any}()
    
    # 1. Random Forest - Robust baseline
    models["RandomForest"] = RandomForestClassifier(
        n_estimators = 100,
        max_depth = 10,
        min_samples_split = 5,
        min_samples_leaf = 2,
        random_state = 42,
        n_jobs = -1  # Use all CPU cores
    )
    
    # 2. Decision Tree - Fast and interpretable
    models["DecisionTree"] = DecisionTreeClassifier(
        max_depth = 8,
        min_samples_split = 10,
        min_samples_leaf = 5,
        random_state = 42
    )
    
    # 3. XGBoost - High performance
    models["XGBoost"] = XGBoostClassifier(
        max_depth = 6,
        n_estimators = 100,
        learning_rate = 0.1,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42,
        n_jobs = -1
    )
    
    # 4. Multinomial Naive Bayes - Good for text features
    models["MultinomialNB"] = MultinomialNB(
        alpha = 1.0,  # Smoothing parameter
        fit_prior = true
    )
    
    # 5. Support Vector Classifier - High-dimensional data
    models["SVC"] = SVC(
        C = 1.0,
        kernel = "rbf",
        gamma = "scale",
        probability = true,  # Enable probability estimates
        random_state = 42
    )
    
    return models
end

# Model evaluation for behavioral intent data
function evaluate_model_ensemble(X, y; cv_folds=5, test_size=0.2)
    """
    Evaluate all starting models on behavioral intent data
    Returns performance metrics and model rankings
    """
    
    models = create_starting_models()
    results = Dict{String, Dict{String, Float64}}()
    
    # Split data
    train, test = partition(eachindex(y), test_size, rng=MersenneTwister(42))
    X_train, X_test = X[train, :], X[test, :]
    y_train, y_test = y[train], y[test]
    
    println("🔬 Evaluating $(length(models)) models on behavioral data...")
    println("Training samples: $(length(train)), Test samples: $(length(test))")
    
    for (name, model) in models
        println("\n📊 Evaluating $name...")
        
        try
            # Create machine
            mach = machine(model, X_train, y_train)
            
            # Cross-validation
            cv_result = evaluate!(mach, 
                                resampling=CV(nfolds=cv_folds, rng=42),
                                measure=[accuracy, f1score, log_loss],
                                verbosity=0)
            
            # Test performance
            fit!(mach, verbosity=0)
            y_pred = predict_mode(mach, X_test)
            y_pred_proba = predict(mach, X_test)
            
            test_acc = accuracy(y_pred, y_test)
            test_f1 = f1score(y_pred, y_test)
            
            # Store results
            results[name] = Dict(
                "cv_accuracy" => mean(cv_result.measurement[1]),
                "cv_accuracy_std" => std(cv_result.measurement[1]),
                "cv_f1score" => mean(cv_result.measurement[2]),
                "cv_f1score_std" => std(cv_result.measurement[2]),
                "cv_logloss" => mean(cv_result.measurement[3]),
                "test_accuracy" => test_acc,
                "test_f1score" => test_f1,
                "fit_time" => cv_result.times[1]  # Training time
            )
            
            println("  ✅ CV Accuracy: $(round(results[name]["cv_accuracy"], digits=3)) ± $(round(results[name]["cv_accuracy_std"], digits=3))")
            println("     Test Accuracy: $(round(test_acc, digits=3))")
            println("     Training time: $(round(results[name]["fit_time"], digits=2))s")
            
        catch e
            println("  ❌ Failed: $e")
            results[name] = Dict("error" => string(e))
        end
    end
    
    return results
end

# Select best performing model
function select_best_model(results::Dict{String, Dict{String, Float64}})
    """
    Select best model based on combined performance metrics
    """
    
    valid_results = filter(p -> !haskey(p[2], "error"), results)
    
    if isempty(valid_results)
        error("No valid model results found")
    end
    
    # Score models (higher is better)
    scores = Dict{String, Float64}()
    
    for (name, metrics) in valid_results
        # Weighted score: accuracy (40%) + f1 (40%) + speed (20%)
        acc_score = metrics["cv_accuracy"]
        f1_score = metrics["cv_f1score"]
        
        # Speed score (inverse of time, normalized)
        times = [m["fit_time"] for m in values(valid_results) if haskey(m, "fit_time")]
        min_time = minimum(times)
        speed_score = min_time / metrics["fit_time"]
        
        scores[name] = 0.4 * acc_score + 0.4 * f1_score + 0.2 * speed_score
    end
    
    # Sort by score
    ranked = sort(collect(scores), by=x->x[2], rev=true)
    
    println("\n🏆 Model Rankings:")
    for (i, (name, score)) in enumerate(ranked)
        metrics = results[name]
        println("  $i. $name (score: $(round(score, digits=3)))")
        println("     Accuracy: $(round(metrics["cv_accuracy"], digits=3)), F1: $(round(metrics["cv_f1score"], digits=3)), Time: $(round(metrics["fit_time"], digits=2))s")
    end
    
    best_model = ranked[1][1]
    println("\n✨ Best model: $best_model")
    
    return best_model, ranked
end

# Generate synthetic behavioral intent data for testing
function generate_behavioral_data(n_samples=1000)
    """
    Generate synthetic behavioral intent data for model testing
    """
    
    Random.seed!(42)
    
    # Intent categories (matching your training system)
    intents = ["help", "create", "analyze", "learn", "communicate", 
              "problem_solve", "explore", "optimize"]
    
    # Generate features (matching your 256-dimensional feature vector)
    n_features = 50  # Reduced for demo
    
    X = randn(n_samples, n_features)
    
    # Add structure to make classification meaningful
    for i in 1:n_samples
        intent_idx = rand(1:length(intents))
        
        # Add intent-specific patterns
        if intent_idx == 1  # help
            X[i, 1:5] .+= 2.0  # High "urgency" features
        elseif intent_idx == 2  # create
            X[i, 6:10] .+= 2.0  # High "creativity" features
        elseif intent_idx == 3  # analyze
            X[i, 11:15] .+= 2.0  # High "analytical" features
        elseif intent_idx == 4  # learn
            X[i, 16:20] .+= 2.0  # High "learning" features
        end
        
        # Add some noise
        X[i, :] .+= 0.5 * randn(n_features)
    end
    
    # Generate labels
    y = [rand(intents) for _ in 1:n_samples]
    
    # Convert to DataFrame
    feature_names = ["feature_$i" for i in 1:n_features]
    X_df = DataFrame(X, feature_names)
    
    return X_df, categorical(y)
end

# Integration with behavioral training system
function create_mlj_intent_classifier(model_name::String, X, y)
    """
    Create a trained MLJ classifier for behavioral intent prediction
    """
    
    models = create_starting_models()
    
    if !haskey(models, model_name)
        error("Model $model_name not found. Available: $(keys(models))")
    end
    
    model = models[model_name]
    mach = machine(model, X, y)
    fit!(mach, verbosity=1)
    
    println("✅ $model_name trained on $(nrows(X)) samples")
    
    return mach
end

# Prediction interface for training system
function predict_behavioral_intent(mach, features::Vector{Float64})
    """
    Predict intent from behavioral features using trained MLJ model
    """
    
    # Convert features to DataFrame (assuming same feature names as training)
    n_features = length(features)
    feature_names = ["feature_$i" for i in 1:n_features]
    X_new = DataFrame([features], feature_names)
    
    # Get predictions
    pred_proba = predict(mach, X_new)
    pred_mode = predict_mode(mach, X_new)
    
    # Extract confidence (probability of predicted class)
    confidence = maximum(pdf.(pred_proba[1], levels(pred_proba[1])))
    
    return string(pred_mode[1]), confidence
end

# Demo function
function demo_mlj_models()
    println("🚀 MLJ Models Demo for Behavioral Intent Classification")
    println("=" ^ 65)
    
    # Generate test data
    println("📊 Generating synthetic behavioral data...")
    X, y = generate_behavioral_data(2000)
    println("✅ Generated $(nrows(X)) samples with $(ncols(X)) features")
    
    # Evaluate all models
    results = evaluate_model_ensemble(X, y)
    
    # Select best model
    best_model, rankings = select_best_model(results)
    
    # Train best model
    println("\n🎯 Training best model: $best_model")
    mach = create_mlj_intent_classifier(best_model, X, y)
    
    # Test prediction
    println("\n🔮 Testing prediction interface...")
    test_features = randn(ncols(X))
    intent, confidence = predict_behavioral_intent(mach, test_features)
    println("  Sample prediction: $intent (confidence: $(round(confidence, digits=3)))")
    
    println("\n✅ MLJ models demo complete!")
    
    return mach, results, rankings
end

# Model characteristics summary
function model_characteristics()
    return Dict(
        "RandomForest" => Dict(
            "strengths" => ["Robust to overfitting", "Handles mixed features", "Feature importance"],
            "use_case" => "General purpose, baseline model",
            "speed" => "Medium",
            "interpretability" => "Medium"
        ),
        "DecisionTree" => Dict(
            "strengths" => ["Fast training/prediction", "Highly interpretable", "Rule extraction"],
            "use_case" => "Quick prototyping, rule-based systems",
            "speed" => "Fast",
            "interpretability" => "High"
        ),
        "XGBoost" => Dict(
            "strengths" => ["High accuracy", "Feature selection", "Regularization"],
            "use_case" => "Production systems, competitions",
            "speed" => "Medium-slow",
            "interpretability" => "Low-medium"
        ),
        "MultinomialNB" => Dict(
            "strengths" => ["Fast", "Good with text", "Probabilistic"],
            "use_case" => "Text classification, categorical features",
            "speed" => "Very fast",
            "interpretability" => "Medium"
        ),
        "SVC" => Dict(
            "strengths" => ["High-dimensional data", "Kernel methods", "Robust"],
            "use_case" => "Complex decision boundaries, small datasets",
            "speed" => "Slow",
            "interpretability" => "Low"
        )
    )
end

println("🎯 Five MLJ Starting Models Ready")
println("Use: demo_mlj_models() to test all models")
println("Available models: $(keys(create_starting_models()))")

end # Starting....

