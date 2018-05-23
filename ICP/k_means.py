from sklearn.cluster import KMeans


def k_means():
    labels = []
    train_data = []
    test_data = []
    with open('assets/leuk72_3k.txt', 'r') as f_in:
        count = 0
        while True:
            line = f_in.readline()
            if not line:
                break
            items = line.split('\t')
            labels.append(int(items[0]))
            train_data.append(map(float, items[1:]))
            if count % 5 == 0:
                test_data.append(items[1:])
            count += 1
    kmeans = KMeans(n_clusters=3).fit(train_data)
    results = kmeans.predict(test_data)
    print results

