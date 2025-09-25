import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;


public class read_checkpoint {
    
  
    public static class Config {
        public int dim;          
        public int hidden_dim;   
        public int n_layers;     
        public int n_heads;      
        public int n_kv_heads;   
        public int vocab_size;  
        public int seq_len;      
        
        @Override
        public String toString() {
            return String.format(
                "Config{dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d}",
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            );
        }
    }
    
    
    public static class Matrix {
        public final float[] data;
        public final int rows;
        public final int cols;
        
        public Matrix(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            this.data = new float[rows * cols];
        }
        
        public float get(int row, int col) {
            return data[row * cols + col];
        }
        
        public void set(int row, int col, float value) {
            data[row * cols + col] = value;
        }
        
        public void loadFromBuffer(ByteBuffer buffer, int count) {
            for (int i = 0; i < count; i++) {
                data[i] = buffer.getFloat();
            }
        }
    }
    
    
    public static class TransformerWeights {
        public Matrix token_embedding_table;  
        
        public Matrix rms_att_weight;         
        public Matrix rms_ffn_weight;         
        public float[] rms_final_weight;     
        
        public Matrix[] wq;  
        public Matrix[] wk;  
        public Matrix[] wv;  
        public Matrix[] wo;  
        
        public Matrix[] w1;  
        public Matrix[] w2;  
        public Matrix[] w3;  
        
        public Matrix wcls;
        
        public boolean shared_weights;
    }
    
    /**
     * Read checkpoint file and load configuration and weights
     * 
     * @param checkpointPath Path to the checkpoint file
     * @return Object containing config and weights
     * @throws IOException if file cannot be read
     */
    public static CheckpointData readCheckpoint(String checkpointPath) throws IOException {
        File file = new File(checkpointPath);
        if (!file.exists()) {
            throw new IOException("Checkpoint file does not exist: " + checkpointPath);
        }
        
        Config config = new Config();
        TransformerWeights weights = new TransformerWeights();
        
        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel channel = raf.getChannel()) {
            
            ByteBuffer configBuffer = ByteBuffer.allocate(7 * 4);
            configBuffer.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(configBuffer);
            configBuffer.flip();
            
            config.dim = configBuffer.getInt();
            config.hidden_dim = configBuffer.getInt();
            config.n_layers = configBuffer.getInt();
            config.n_heads = configBuffer.getInt();
            config.n_kv_heads = configBuffer.getInt();
            config.vocab_size = configBuffer.getInt();
            config.seq_len = configBuffer.getInt();
            
            weights.shared_weights = config.vocab_size > 0;
            config.vocab_size = Math.abs(config.vocab_size);
            
            int head_size = config.dim / config.n_heads;
            int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
            
            long weightsSize = channel.size() - channel.position();
            ByteBuffer weightsBuffer = ByteBuffer.allocate((int)weightsSize);
            weightsBuffer.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(weightsBuffer);
            weightsBuffer.flip();
            
            
            weights.token_embedding_table = new Matrix(config.vocab_size, config.dim);
            weights.token_embedding_table.loadFromBuffer(weightsBuffer, config.vocab_size * config.dim);
            
            weights.rms_att_weight = new Matrix(config.n_layers, config.dim);
            weights.rms_att_weight.loadFromBuffer(weightsBuffer, config.n_layers * config.dim);
            
            weights.wq = new Matrix[config.n_layers];
            for (int l = 0; l < config.n_layers; l++) {
                weights.wq[l] = new Matrix(config.dim, config.dim);
                weights.wq[l].loadFromBuffer(weightsBuffer, config.dim * config.dim);
            }
            
            weights.wk = new Matrix[config.n_layers];
            for (int l = 0; l < config.n_layers; l++) {
                weights.wk[l] = new Matrix(config.dim, kv_dim);
                weights.wk[l].loadFromBuffer(weightsBuffer, config.dim * kv_dim);
            }
            
            weights.wv = new Matrix[config.n_layers];
            for (int l = 0; l < config.n_layers; l++) {
                weights.wv[l] = new Matrix(config.dim, kv_dim);
                weights.wv[l].loadFromBuffer(weightsBuffer, config.dim * kv_dim);
            }
            
            weights.wo = new Matrix[config.n_layers];
            for (int l = 0; l < config.n_layers; l++) {
                weights.wo[l] = new Matrix(config.dim, config.dim);
                weights.wo[l].loadFromBuffer(weightsBuffer, config.dim * config.dim);
            }
            
            weights.rms_ffn_weight = new Matrix(config.n_layers, config.dim);
            weights.rms_ffn_weight.loadFromBuffer(weightsBuffer, config.n_layers * config.dim);
            
            weights.w1 = new Matrix[config.n_layers];
            for (int l = 0; l < config.n_layers; l++) {
                weights.w1[l] = new Matrix(config.hidden_dim, config.dim);
                weights.w1[l].loadFromBuffer(weightsBuffer, config.hidden_dim * config.dim);
            }
            
            weights.w2 = new Matrix[config.n_layers];
            for (int l = 0; l < config.n_layers; l++) {
                weights.w2[l] = new Matrix(config.dim, config.hidden_dim);
                weights.w2[l].loadFromBuffer(weightsBuffer, config.dim * config.hidden_dim);
            }
            
            weights.w3 = new Matrix[config.n_layers];
            for (int l = 0; l < config.n_layers; l++) {
                weights.w3[l] = new Matrix(config.hidden_dim, config.dim);
                weights.w3[l].loadFromBuffer(weightsBuffer, config.hidden_dim * config.dim);
            }
            
            weights.rms_final_weight = new float[config.dim];
            for (int i = 0; i < config.dim; i++) {
                weights.rms_final_weight[i] = weightsBuffer.getFloat();
            }
            
            weightsBuffer.position(weightsBuffer.position() + config.seq_len * head_size * 4);
            
            if (weights.shared_weights) {
                weights.wcls = weights.token_embedding_table;
            } else {
                weights.wcls = new Matrix(config.vocab_size, config.dim);
                weights.wcls.loadFromBuffer(weightsBuffer, config.vocab_size * config.dim);
            }
        }
        
        return new CheckpointData(config, weights);
    }
    
  
    public static class CheckpointData {
        public final Config config;
        public final TransformerWeights weights;
        
        public CheckpointData(Config config, TransformerWeights weights) {
            this.config = config;
            this.weights = weights;
        }
    }
    
 
    public static void main(String[] args) {
        System.out.println("Test 1: Loading checkpoint and verifying configuration");
        System.out.println("======================================================");
        
        try {
            CheckpointData checkpoint = readCheckpoint("stories15M.bin");
            Config config = checkpoint.config;
            
            System.out.println("Configuration loaded successfully!");
            System.out.println(config);
            
            assert config.dim == 288 : "Expected dim=288, got " + config.dim;
            assert config.hidden_dim == 768 : "Expected hidden_dim=768, got " + config.hidden_dim;
            assert config.n_layers == 6 : "Expected n_layers=6, got " + config.n_layers;
            assert config.n_heads == 6 : "Expected n_heads=6, got " + config.n_heads;
            assert config.n_kv_heads == 6 : "Expected n_kv_heads=6, got " + config.n_kv_heads;
            assert config.vocab_size == 32000 : "Expected vocab_size=32000, got " + config.vocab_size;
            assert config.seq_len == 256 : "Expected seq_len=256, got " + config.seq_len;
            
            System.out.println("✓ Config validation passed!");
            
        } catch (IOException e) {
            System.err.println("Failed to load checkpoint: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\nTest 2: Verifying weight dimensions");
        System.out.println("====================================");
        
        try {
            CheckpointData checkpoint = readCheckpoint("stories15M.bin");
            TransformerWeights weights = checkpoint.weights;
            Config config = checkpoint.config;
            
            assert weights.token_embedding_table.rows == config.vocab_size;
            assert weights.token_embedding_table.cols == config.dim;
            System.out.println("✓ Token embedding table: " + 
                weights.token_embedding_table.rows + " x " + 
                weights.token_embedding_table.cols);
            
            assert weights.wq.length == config.n_layers;
            assert weights.wk.length == config.n_layers;
            assert weights.wv.length == config.n_layers;
            assert weights.wo.length == config.n_layers;
            System.out.println("✓ Attention weights: " + config.n_layers + " layers");
            
            assert weights.w1.length == config.n_layers;
            assert weights.w2.length == config.n_layers;
            assert weights.w3.length == config.n_layers;
            System.out.println("✓ FFN weights: " + config.n_layers + " layers");
            
            assert weights.wq[0].rows == config.dim;
            assert weights.wq[0].cols == config.dim;
            System.out.println("✓ Query weight dimensions: " + 
                weights.wq[0].rows + " x " + weights.wq[0].cols);
            
            int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
            assert weights.wk[0].cols == kv_dim;
            assert weights.wv[0].cols == kv_dim;
            System.out.println("✓ Key/Value dimensions: " + kv_dim);
            
            System.out.println("✓ All weight dimensions verified!");
            
        } catch (IOException e) {
            System.err.println("Failed to load checkpoint: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\nTest 3: Sampling weight values");
        System.out.println("===============================");
        
        try {
            CheckpointData checkpoint = readCheckpoint("stories15M.bin");
            TransformerWeights weights = checkpoint.weights;
            
            System.out.println("First 5 values from token embedding[0]:");
            for (int i = 0; i < 5; i++) {
                System.out.printf("  [%d]: %.6f\n", i, weights.token_embedding_table.data[i]);
            }
            
            System.out.println("\nFirst 3 values from wq[0]:");
            for (int i = 0; i < 3; i++) {
                System.out.printf("  [%d]: %.6f\n", i, weights.wq[0].data[i]);
            }
            
            System.out.println("\nWeights shared with embeddings: " + weights.shared_weights);
            
            System.out.println("✓ Weight sampling complete!");
            
        } catch (IOException e) {
            System.err.println("Failed to load checkpoint: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n All tests completed successfully!");
    }
}
