import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import TimeSeriesSplit  # this ended up being important for better validation

# i will use this to make time metrics
def generate_metric(length=1000):
    np.random.seed(888) # this is just for consistency, idk i just like number 8

    metric = []      # in this list i will store metric value at each time step
    incidents = []   # and here whether or not incident happened or not, so this is just ones and zeros
# 0 = normal
# 1 = incident

    value = 50       # some baseline value
    incident_duration = 0  # this ended up being important, i use it to track how long current incident lasts

    for i in range(length):  # each iteration is basically one moment in time
        noise = np.random.normal(0, 1) # generate a single random number from a standard normal distribution (mean=0, std=1), so like around 68% of the time i get something between -1 and 1
        value = value + noise  # i slowly change the metric over time

        # this is to handle incident duration
        if incident_duration > 0:
            # i know we are in an ongoing incident
            incident = 1
            value = value + np.random.uniform(10, 20)  # keep value elevated
            incident_duration -= 1
        else:
            incident = 0  # i start assuming nothing bad happened
            # maybe start a new incident - increased probability to get more incidents
            if np.random.rand() < 0.08:  # rarely to cause it
                incident_duration = np.random.randint(5, 12)  # longer incidents (5-11 steps)
                value = value + np.random.uniform(20, 35)  # and larger spikes
                incident = 1
                incident_duration -= 1

        metric.append(value)
        incidents.append(incident)

    metric = np.array(metric)
    incidents = np.array(incidents)

    return metric, incidents

# now even though i have metrics and incidents lists, i still need data in form that is useful for training the model
# so something like [49.8, 50.3, 50.5, 50.2, 50.0] -> 1 meaning for these past metrics the incident will happen
# X will be these features, and Y will be for labels
def create_sliding_windows(metric, incidents, window_size, horizon):

    X = []  # so this will store past windows
    y = []  # and this will be just the list for labels, 0 if normal, 1 if incident

    total_length = len(metric)

    # this might be a bit confusing, but the idea was just not to run out of array
    # basically i have to start at window size becuase that is how far back i want to look, and end at total length - horizon but that how far ahead i want to look
    for t in range(window_size, total_length - horizon):

        # these are past metrics i take based on that window size
        past_window = metric[t-window_size:t]

        # also include statistical features that summarize the windows behavior
        window_mean = np.mean(past_window)
        window_std = np.std(past_window)
        window_max = np.max(past_window)
        window_min = np.min(past_window)
        recent_trend = past_window[-1] - past_window[-5] if window_size >= 5 else 0
        above_threshold = np.sum(past_window > np.mean(past_window) + 2*np.std(past_window))
        
        # combine raw window with statistics
        combined_features = np.concatenate([past_window, 
                                          [window_mean, window_std, window_max, window_min, recent_trend, above_threshold]])

        # here i look for if there are any incidents on the horizon
        future_incidents = incidents[t:t+horizon]

        label = 1 if np.any(future_incidents == 1) else 0   # if there was an incident i set it to 1, if not than 0

        # and then i add them here, so now i have data in form useful for training
        X.append(combined_features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    

    return X, y


# okay, this is basic data split of 80% for training, 20% for testing
# usually i would shuffle, but that should not be done here because time sequence is relevant
def split_train_test(X, y, train_ratio=0.8):

    total_samples = len(X)
    split_index = int(total_samples * train_ratio)

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


# for this type of problem i will obviously be using the binary classification model 
def train_model(X_train, y_train):

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")  
    model.fit(X_train_scaled, y_train)

    return model, scaler


# with this i check how good the model is
def evaluate_model(model, scaler, X_test, y_test, threshold=0.5):

    # scale test data
    X_test_scaled = scaler.transform(X_test)

    # so i get probabilities that incident will occur (which is column 1)
    probabilities = model.predict_proba(X_test_scaled)[:,1]

    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = threshold
    
    for thresh in thresholds:
        preds = (probabilities >= thresh).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"Using optimal threshold: {best_threshold:.3f}")
    
    # because these need to be binary predictions i need to make it basically a yes or no by using the threshold, and then i also convert it with astype(int) from True/False into 1/0
    predictions = (probabilities >= best_threshold).astype(int)

    # this calculates the precision, so basically how many true positives are there out of total number of predictions. P = TP/(TP + FP)
    precision = precision_score(y_test, predictions, zero_division=0)  # this is to avoid warning

    # and this is for figuring out, out of all actual incidents, how many did the model detect. Just R = TP/(TP+FN), if recall is high that means model missed fewer incidents
    recall = recall_score(y_test, predictions, zero_division=0)

    # funny that there is a function for this as well but basically F1 = 2*(precision * recall)/(precision + recall)
    # basically a middleground for both
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return probabilities 


# save model weights to json
def save_model_to_json(model, scaler, filename="model.json"):

    model_data = {
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),  # fancy way to say bias but okay
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }

    with open(filename, "w") as f:
        json.dump(model_data, f)

    print("Yay, model saved.")


if __name__ == "__main__":
    metric, incidents = generate_metric(5000)
    X, y = create_sliding_windows(metric, incidents, window_size=20, horizon=5)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model, scaler = train_model(X_train, y_train)
    evaluate_model(model, scaler, X_test, y_test)
    save_model_to_json(model, scaler)