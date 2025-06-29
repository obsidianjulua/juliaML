 # =============================================================================
    # PLUTO NOTEBOOK TEMPLATE FOR METAMORPHIC ML
    # Save as metamorphic_template.jl and open with Pluto
    # =============================================================================

    ### A Pluto.jl notebook ###
    # v0.19.40

    using Markdown
    using InteractiveUtils

    # ╔═╡ Cell order:
    # ╟─metamorphic_setup
    # ╟─model_creation
    # ╟─injection_interface
    # ╟─visualization_panel
    # ╟─performance_monitor
    # ╟─experiment_log

    # ╔═╡ metamorphic_setup ╠═╡
    begin
        using Pkg
        Pkg.activate(".")

        using Flux, MLJ, Transformers
        using Plots, PlutoUI, HypertextLiteral
        using LinearAlgebra, Statistics
        using BenchmarkTools

        # Load your metamorphic framework
        include("metamorphic_injection.jl")

        md"## 🧬 MetaMorphic ML Interactive Workspace"
    end

    # ╔═╡ model_creation ╠═╡
    begin
        # Base model creation
        @bind model_type Select([
            "transformer" => "Transformer Block",
            "dense_chain" => "Dense Chain",
            "custom" => "Custom Architecture"
            ])

        base_model = if model_type == "transformer"
            Chain(
                Dense(128, 64, relu),
                TransformerBlock(8, 64),  # 8 heads, 64 dim
                Dense(64, 32, relu),
                Dense(32, 10)
                )
            elseif model_type == "dense_chain"
            Chain(
                Dense(128, 64, relu),
                Dense(64, 32, relu),
                Dense(32, 10)
                )
        else
            # Custom model input
            Chain(Dense(128, 10))
        end

        # Convert to metamorphic architecture
        meta_model = build_constructivist_architecture(
            base_model,
            Dict(:target_types => [Dense, TransformerBlock])
            )

        HTML("""
             <div style="background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
             padding: 15px; border-radius: 10px; color: white;">
             <h3>🏗️ Model Architecture</h3>
             <p>Base Model: $(typeof(base_model))</p>
             <p>Layers: $(length(base_model.layers))</p>
             <p>MetaMorphic Layers: $(count(x -> x isa MetaMorphicLayer, meta_model.layers))</p>
             </div>
             """)
    end

    # ╔═╡ injection_interface ╠═╡
    begin
        md"### 💉 Logic Injection Interface"

        @bind injection_text TextField(default="optimize attention and boost performance")

        @bind break_points MultiCheckBox([
            :attention_redirect => "Attention Redirect",
            :logic_bypass => "Logic Bypass",
            :pattern_override => "Pattern Override",
            :optimization_boost => "Optimization Boost"
            ])

        @bind inject_button Button("🚀 Apply Metamorphic Injection")
    end

    # ╔═╡ injection_execution ╠═╡
    begin
        inject_button  # Trigger on button press

        # Apply constructivist training session
        if inject_button > 0
            global modified_model, detected_patterns = constructivist_training_session!(
                deepcopy(meta_model),
                injection_text,
                collect(break_points),
                Dict(:boost_factor => 1.3, :visualization => true)
                )

            @info "Injection applied" patterns=detected_patterns modifications=length(break_points)
        else
            global modified_model = meta_model
            global detected_patterns = Symbol[]
        end

        HTML("""
             <div style="background: #2d3748; padding: 15px; border-radius: 10px; color: #e2e8f0;">
             <h4>🔬 Injection Results</h4>
             <p><strong>Detected Patterns:</strong> $(join(string.(detected_patterns), ", "))</p>
             <p><strong>Active Break Points:</strong> $(join(string.(break_points), ", "))</p>
             <p><strong>Injection Status:</strong> $(inject_button > 0 ? "✅ Applied" : "⏳ Pending")</p>
             </div>
             """)
    end

    # ╔═╡ visualization_panel ╠═╡
    begin
        md"### 📊 Real-time Visualization"

        # Test input for visualization
        test_input = randn(Float32, 128, 32)  # batch of 32

        if @isdefined modified_model
            # Run inference and capture intermediate outputs
            original_output = base_model(test_input)
            modified_output = modified_model(test_input)

            # Create comparison plots
            p1 = heatmap(original_output[1:min(20, size(original_output, 1)), 1:min(10, size(original_output, 2))],
                         title="Original Output", c=:viridis)

            p2 = heatmap(modified_output[1:min(20, size(modified_output, 1)), 1:min(10, size(modified_output, 2))],
                         title="MetaMorphic Output", c=:plasma)

            # Difference analysis
            diff_matrix = abs.(original_output - modified_output)
            p3 = heatmap(diff_matrix[1:min(20, size(diff_matrix, 1)), 1:min(10, size(diff_matrix, 2))],
                         title="Absolute Difference", c=:hot)

            # Performance metrics
            similarity_score = cor(vec(original_output), vec(modified_output))
            magnitude_change = norm(modified_output) / norm(original_output)

            p4 = bar(["Similarity", "Magnitude Ratio"],
                     [similarity_score, magnitude_change],
                     title="Metamorphic Metrics",
                     ylim=(0, 2), color=[:blue, :red])

            plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
        else
            plot(rand(10, 10), title="Awaiting Model Injection...", c=:grays)
        end
    end

    # ╔═╡ performance_monitor ╠═╡
    begin
        md"### ⚡ Performance Monitor"

        if @isdefined modified_model
            # Benchmark comparison
            original_time = @belapsed $base_model($test_input)
            modified_time = @belapsed $modified_model($test_input)

            # Memory usage (approximate)
            original_params = sum(length, Flux.params(base_model))
            modified_params = sum(length, Flux.params(modified_model))

            performance_data = [
                ("Inference Time (ms)", original_time * 1000, modified_time * 1000),
                ("Parameter Count", original_params, modified_params),
                ("Memory Ratio", 1.0, modified_params / original_params)
                ]

            HTML("""
                 <div style="background: #1a202c; color: #e2e8f0; padding: 20px; border-radius: 10px; font-family: monospace;">
                 <h4>📈 Performance Comparison</h4>
                 <table style="width: 100%; border-collapse: collapse;">
                 <tr style="border-bottom: 1px solid #4a5568;">
                 <th style="text-align: left; padding: 8px;">Metric</th>
                 <th style="text-align: right; padding: 8px;">Original</th>
                 <th style="text-align: right; padding: 8px;">MetaMorphic</th>
                 <th style="text-align: right; padding: 8px;">Ratio</th>
                 </tr>
                 $(join(["""
                         <tr>
                         <td style="padding: 8px;">$(metric)</td>
                         <td style="padding: 8px; text-align: right;">$(round(orig, digits=3))</td>
                         <td style="padding: 8px; text-align: right;">$(round(mod, digits=3))</td>
                         <td style="padding: 8px; text-align: right; color: $(mod/orig > 1.1 ? "#fc8181" : mod/orig < 0.9 ? "#68d391" : "#a0aec0");">
                         $(round(mod/orig, digits=3))x
                         </td>
                         </tr>
                         """ for (metric, orig, mod) in performance_data]))
                 </table>
                 </div>
                 """)
                         else
                             HTML("<p>Run injection to see performance metrics...</p>")
                         end
        end

        # ╔═╡ experiment_log ╠═╡
        begin
            md"### 📝 Experiment Log"

            # Log of all experiments
            if !@isdefined experiment_history
                global experiment_history = []
            end

            # Add current experiment if injection was applied
            if inject_button > 0 && @isdefined modified_model
                experiment_entry = (
                    timestamp = now(),
                    injection_text = injection_text,
                    break_points = collect(break_points),
                    patterns = detected_patterns,
                    performance_ratio = @isdefined(modified_time) ? modified_time / original_time : 1.0
                    )

                # Only add if not duplicate
                if isempty(experiment_history) || experiment_history[end].timestamp != experiment_entry.timestamp
                    push!(experiment_history, experiment_entry)
                end
            end

            # Display experiment history
            if !isempty(experiment_history)
                HTML("""
                     <div style="background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 10px; max-height: 300px; overflow-y: auto;">
                     <h4>🔬 Experiment History</h4>
                     $(join(["""
                             <div style="border: 1px solid #4a5568; margin: 5px 0; padding: 10px; border-radius: 5px;">
                             <strong>$(exp.timestamp)</strong><br>
                             <em>Input:</em> "$(exp.injection_text)"<br>
                             <em>Break Points:</em> $(join(string.(exp.break_points), ", "))<br>
                             <em>Patterns:</em> $(join(string.(exp.patterns), ", "))<br>
                             <em>Performance:</em> $(round(exp.performance_ratio, digits=3))x
                             </div>
                             """ for exp in reverse(experiment_history[max(1, end-4):end])]))
                     </div>
                     """)
                             else
                                 HTML("<p><em>No experiments logged yet. Apply an injection to start logging.</em></p>")
                             end
            end

            # ╔═╡ advanced_controls ╠═╡
            begin
                md"### 🛠️ Advanced Controls"

                with_terminal() do
                    if @isdefined modified_model
                        println("🧬 MetaMorphic Model Analysis")
                        println("=" ^ 40)

                        # Count metamorphic layers
                        metamorphic_count = 0
                        total_modifications = 0

                        for layer in modified_model.layers
                            if layer isa MetaMorphicLayer
                                metamorphic_count += 1
                                total_modifications += length(layer.active_modifications)
                                println("Layer $(metamorphic_count): $(length(layer.active_modifications)) modifications")
                            end
                        end

                        println("\nSummary:")
                        println("  MetaMorphic Layers: $(metamorphic_count)")
                        println("  Total Modifications: $(total_modifications)")
                        println("  Injection Efficiency: $(round(total_modifications/max(1,metamorphic_count), digits=2)) modifications/layer")

                        if haskey(METAMORPHIC_STATS, :injection)
                            avg_injection_time = mean(METAMORPHIC_STATS[:injection]) * 1000
                            println("  Avg Injection Time: $(round(avg_injection_time, digits=2)) ms")
                        end
                    else
                        println("⏳ Awaiting model injection...")
                    end
                end
            end
