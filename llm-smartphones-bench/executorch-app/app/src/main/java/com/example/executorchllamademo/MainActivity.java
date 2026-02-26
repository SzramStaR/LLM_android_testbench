/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import android.app.ActivityManager;
import android.app.AlertDialog;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.system.ErrnoException;
import android.system.Os;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ListView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CustomBenchmarkRunner.BenchmarkCallback {
  private ListView mModelsListView;
  private ModelListAdapter mModelListAdapter;
  private List<BenchmarkModel> mModels;
  private ImageButton mSettingsButton;
  private TextView mMemoryView;
  private DemoSharedPreferences mDemoSharedPreferences;
  private Handler mMemoryUpdateHandler;
  private Runnable memoryUpdater;
  private CustomBenchmarkRunner mCurrentBenchmarkRunner = null;


  @Override
  public void onBenchmarkStarted(String modelName) {
    runOnUiThread(() -> {
      updateModelStatus(modelName, BenchmarkModel.BenchmarkStatus.LOADING);
      ETLogging.getInstance().log("Benchmark started for model: " + modelName);
    });
  }

  @Override
  public void onModelLoaded(String modelName, boolean success, long loadTimeMs) {
    runOnUiThread(() -> {
      if (success) {
        updateModelStatus(modelName, BenchmarkModel.BenchmarkStatus.RUNNING);
        ETLogging.getInstance().log("Model loaded successfully: " + modelName + " in " + loadTimeMs + "ms");
      } else {
        updateModelStatus(modelName, BenchmarkModel.BenchmarkStatus.ERROR);
        ETLogging.getInstance().log("Model loading failed: " + modelName);
      }
    });
  }

  @Override
  public void onInferenceStarted(String modelName, int inputTokens, int runNumber, int totalRuns) {
    runOnUiThread(() -> {
      updateModelStatus(modelName, BenchmarkModel.BenchmarkStatus.RUNNING);
      String logMessage = String.format("Inference started for %s: %d tokens (run %d/%d)", 
          modelName, inputTokens, runNumber, totalRuns);
      ETLogging.getInstance().log(logMessage);
    });
  }

  @Override
  public void onInferenceCompleted(String modelName, int inputTokens, String result, long inferenceTimeMs, double tokensPerSecond, int runNumber, int totalRuns) {
    runOnUiThread(() -> {
      String logMessage = String.format("Inference completed for %s: %d tokens, %dms, %.2f tokens/sec (run %d/%d)", 
          modelName, inputTokens, inferenceTimeMs, tokensPerSecond, runNumber, totalRuns);
      ETLogging.getInstance().log(logMessage);
    });
  }
  
  @Override
  public void onMetricsPosted(String modelName, int inputTokens, int runNumber, int totalRuns, boolean success) {
    runOnUiThread(() -> {
      String status = success ? "posted successfully" : "failed to post";
      String logMessage = String.format("Metrics %s for %s: %d tokens (run %d/%d)", 
          status, modelName, inputTokens, runNumber, totalRuns);
      ETLogging.getInstance().log(logMessage);
    });
  }
  
  @Override
  public void onBenchmarkCompleted(String modelName, int totalRuns) {
    runOnUiThread(() -> {
      updateModelStatus(modelName, BenchmarkModel.BenchmarkStatus.COMPLETED);
      String logMessage = String.format("Benchmark completed for %s: %d total runs", modelName, totalRuns);
      ETLogging.getInstance().log(logMessage);
    });
  }

  @Override
  public void onResourcesReleased(String modelName) {
    runOnUiThread(() -> {
      mCurrentBenchmarkRunner = null;
      ETLogging.getInstance().log("Resources released for model: " + modelName);
    });
  }

  @Override
  public void onBenchmarkError(String modelName, String error) {
    runOnUiThread(() -> {
      updateModelStatus(modelName, BenchmarkModel.BenchmarkStatus.ERROR);
      mCurrentBenchmarkRunner = null;
      ETLogging.getInstance().log("Benchmark error for " + modelName + ": " + error);
      
      AlertDialog.Builder builder = new AlertDialog.Builder(this);
      builder.setTitle("Benchmark Error")
             .setMessage("Error running benchmark for " + modelName + ":\n" + error)
             .setPositiveButton("OK", null)
             .show();
    });
  }

  private void updateModelStatus(String modelName, BenchmarkModel.BenchmarkStatus status) {
    for (BenchmarkModel model : mModels) {
      if (model.getFilename().equals(modelName)) {
        model.setStatus(status);
        mModelListAdapter.updateModel(model);
        break;
      }
    }
  }

  private void runBenchmark(BenchmarkModel model) {
    if (mCurrentBenchmarkRunner != null) {

      AlertDialog.Builder builder = new AlertDialog.Builder(this);
      builder.setTitle("Benchmark In Progress")
             .setMessage("Another benchmark is currently running. Please wait for it to complete.")
             .setPositiveButton("OK", null)
             .show();
      return;
    }



    String modelPath = "/data/local/tmp/llama/" + model.getFilename();

    String tokenizerPath = "/data/local/tmp/llama/" + model.getTokenizerPath();

    
    ETLogging.getInstance().log("Starting benchmark for model: " + model.getFilename());
    ETLogging.getInstance().log("Model path: " + modelPath);
    ETLogging.getInstance().log("Tokenizer path: " + tokenizerPath);
    
    mCurrentBenchmarkRunner = new CustomBenchmarkRunner(
        this,
        modelPath, 
        tokenizerPath, 
        model.getFilename(), 
        this
    );
    
    mCurrentBenchmarkRunner.runBenchmark();
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    if (Build.VERSION.SDK_INT >= 21) {
      getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.status_bar));
      getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.nav_bar));
    }

    try {
      Os.setenv("ADSP_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir, true);
      Os.setenv("LD_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir, true);
    } catch (ErrnoException e) {
      finish();
    }


    mModelsListView = requireViewByIdCompat(R.id.models_list_view);
    mModels = new ArrayList<>();
    

    BenchmarkModel[] defaultModels = BenchmarkModel.getDefaultModels();
    for (BenchmarkModel model : defaultModels) {
      mModels.add(model);
    }
    
    mModelListAdapter = new ModelListAdapter(this, mModels);
    mModelsListView.setAdapter(mModelListAdapter);
    

    mModelListAdapter.setBenchmarkClickListener(this::runBenchmark);
    
    mDemoSharedPreferences = new DemoSharedPreferences(this.getApplicationContext());
    
    mSettingsButton = requireViewByIdCompat(R.id.settings);
    mSettingsButton.setOnClickListener(
        view -> {
          Intent myIntent = new Intent(MainActivity.this, SettingsActivity.class);
          MainActivity.this.startActivity(myIntent);
        });

    mMemoryUpdateHandler = new Handler(Looper.getMainLooper());
    startMemoryUpdate();
    setupShowLogsButton();
  }

  @Override
  protected void onPause() {
    super.onPause();
  }

  @Override
  protected void onResume() {
    super.onResume();

    for (BenchmarkModel model : mModels) {
      if (model.getStatus() != BenchmarkModel.BenchmarkStatus.RUNNING) {
        model.setStatus(BenchmarkModel.BenchmarkStatus.READY);
      }
    }
    mModelListAdapter.notifyDataSetChanged();
  }

  private void setupShowLogsButton() {
    ImageButton showLogsButton = requireViewByIdCompat(R.id.showLogsButton);
    showLogsButton.setOnClickListener(
        view -> {
          Intent myIntent = new Intent(MainActivity.this, LogsActivity.class);
          MainActivity.this.startActivity(myIntent);
        });
  }

  private String updateMemoryUsage() {
    ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
    ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
    if (activityManager == null) {
      return "---";
    }
    activityManager.getMemoryInfo(memoryInfo);
    long totalMem = memoryInfo.totalMem / (1024 * 1024);
    long availableMem = memoryInfo.availMem / (1024 * 1024);
    long usedMem = totalMem - availableMem;
    return usedMem + "MB";
  }

  private void startMemoryUpdate() {
    mMemoryView = requireViewByIdCompat(R.id.ram_usage_live);
    memoryUpdater =
        new Runnable() {
          @Override
          public void run() {
            mMemoryView.setText(updateMemoryUsage());
            mMemoryUpdateHandler.postDelayed(this, 1000);
          }
        };
    mMemoryUpdateHandler.post(memoryUpdater);
  }

  private <T extends View> T requireViewByIdCompat(int viewId) {
    T view = findViewById(viewId);
    if (view == null) {
      throw new IllegalStateException("Missing required view with ID: " + viewId);
    }
    return view;
  }

  @Override
  public void onBackPressed() {
    super.onBackPressed();

    if (mCurrentBenchmarkRunner != null) {
      mCurrentBenchmarkRunner.stop();
      mCurrentBenchmarkRunner = null;
    }
    finish();
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    mMemoryUpdateHandler.removeCallbacks(memoryUpdater);
    
    if (mCurrentBenchmarkRunner != null) {
      mCurrentBenchmarkRunner.stop();
      mCurrentBenchmarkRunner = null;
    }
    
    ETLogging.getInstance().saveLogs();
  }
}
