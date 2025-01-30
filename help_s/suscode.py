BEGIN
    // 真の解を取得
    X_true ← compute_true_solution()

    // 各分割数nについてシミュレーション
    FOR each n in n_values DO
        dt ← T / n
        Initialize error_list

        FOR trial FROM 1 TO trials DO
            // Wiener過程の増分を生成
            dW ← generate_normal_random() * sqrt(dt)
            
            // Euler法でYパスを計算し、Xに変換
            Y ← euler_method(Y0, dt, dW)
            X_euler ← Y_to_X(Y)
            
            // 誤差を収集
            error ← |X_euler - X_true| / |X_true|
            error_list.APPEND(error)
        END FOR
        
        // 平均誤差を記録
        mean_error ← MEAN(error_list)
        mean_error_list.APPEND(mean_error)
    END FOR
    
    // 結果を表示またはプロット
    display_or_plot_results(mean_error_list)
END
