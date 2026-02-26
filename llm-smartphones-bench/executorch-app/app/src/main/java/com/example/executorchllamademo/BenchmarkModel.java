/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

public class BenchmarkModel {
    private String filename;
    private String architecture;
    private String tokenizerPath;
    private BenchmarkStatus status;
    
    public enum BenchmarkStatus {
        READY,
        LOADING,
        RUNNING,
        COMPLETED,
        ERROR
    }
    
    public BenchmarkModel(String filename, String architecture, String tokenizerPath) {
        this.filename = filename;
        this.architecture = architecture;
        this.tokenizerPath = tokenizerPath;
        this.status = BenchmarkStatus.READY;
    }
    
    public String getFilename() {
        return filename;
    }
    
    public String getArchitecture() {
        return architecture;
    }
    
    public String getTokenizerPath() {
        return tokenizerPath;
    }
    
    public BenchmarkStatus getStatus() {
        return status;
    }
    
    public void setStatus(BenchmarkStatus status) {
        this.status = status;
    }
    
    public String getStatusText() {
        switch (status) {
            case READY:
                return "Ready";
            case LOADING:
                return "Loading...";
            case RUNNING:
                return "Running benchmark...";
            case COMPLETED:
                return "Completed";
            case ERROR:
                return "Error";
            default:
                return "Unknown";
        }
    }
    
    public int getStatusColor() {
        switch (status) {
            case READY:
                return 0xFF007700; // Green
            case LOADING:
            case RUNNING:
                return 0xFF0066CC; // Blue
            case COMPLETED:
                return 0xFF006600; // Dark green
            case ERROR:
                return 0xFFCC0000; // Red
            default:
                return 0xFF666666; // Gray
        }
    }
    
    public static BenchmarkModel[] getDefaultModels() {
        return new BenchmarkModel[] {
            new BenchmarkModel(
                "llama3_2_3B_bf16.pte", 
                "LLAMA 3.2 ARCHITECTURE", 
                "tokenizer.bin"
            ),
            new BenchmarkModel(
                "llama3_2_3B_spinquant.pte", 
                "LLAMA 3.2 ARCHITECTURE", 
                "tokenizer.bin"
            ),
            new BenchmarkModel(
                "llama3_1_8b_bf16.pte", 
                "LLAMA 3.1 ARCHITECTURE", 
                "tokenizer.bin"
            ),
            new BenchmarkModel(
                "llama3_1_8b_spinquant.pte", 
                "LLAMA 3.1 ARCHITECTURE", 
                "tokenizer.bin"
            ),
            new BenchmarkModel(
                "smollm2.pte",
                "SMOL architecture",
                "tokenizer.json"
            )
        };
    }
}
