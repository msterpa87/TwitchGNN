from spektral.layers import GCNConv
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

class GNNNodeClassifier(Model):
    def __init__(self, n_labels=3, hidden_channels=64, use_dropout=False, dropout_rate=0.2, l2_reg=0.01):
        super(GNNNodeClassifier, self).__init__()
        self.conv1 = GCNConv(hidden_channels, activation='relu')
        self.conv2 = GCNConv(hidden_channels, activation='relu')
        self.conv3 = GCNConv(hidden_channels, activation='relu')
        self.fn1 = Dense(hidden_channels, activation='relu')
        self.fn2 = Dense(n_labels, activation='softmax')
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
    
    def call(self, inputs):
        x, adj = inputs
        
        x = self.conv1([x, adj])
        x = self.conv2([x, adj])
        x = self.conv3([x, adj])
        
        if self.use_dropout:
            x = Dropout(self.dropout_rate)(x)

        x = self.fn1(x)
        
        if self.use_dropout:
            x = Dropout(self.dropout_rate)(x)
        
        x = self.fn2(x)

        return x

class BasicGCN(Model):
    def __init__(self, n_labels=3, hidden_channels=64):
        super(BasicGCN, self).__init__()
        self.conv1 = GCNConv(hidden_channels, activation='relu')
        self.conv2 = GCNConv(n_labels, activation='softmax')
    
    def call(self, inputs):
        x, adj = inputs
        
        x = self.conv1([x, adj])
        x = self.conv2([x, adj])
        
        return x