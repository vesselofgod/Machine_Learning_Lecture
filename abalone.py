import numpy as np
import csv
import time

np.random.seed(1234)
def randomize(): np.random.seed(time.time())

#하이퍼 파라미터 튜닝
RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001



def abalone_exec(epoch_count=10, mb_size=10, report=1):
    '''main function'''
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


def load_abalone_dataset():
    '''데이터파일의 내용을 메모리로 읽어들인 후 rows라는 리스트에 데이터 저장.'''
    with open('abalone.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)#첫행은 건너뜀.
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 10, 1#입출력 벡터 크기 조절
    data = np.zeros([len(rows), input_cnt + output_cnt])

    #성별의 one hot vector expression
    for n, row in enumerate(rows):
        if row[0] == 'I': data[n, 0] = 1
        if row[0] == 'M': data[n, 1] = 1
        if row[0] == 'F': data[n, 2] = 1
        #성별 이외의 정보를 data행렬로 복사함.
        data[n, 3:] = row[1:]


def init_model():
    '''initalization weight and bias'''

    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD,[input_cnt, output_cnt])
    bias = np.zeros([output_cnt])


def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)#data를 섞고, training set과 test set을 분리함.
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        losses, accs = [], []

        for n in range(step_count):
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)
        #epoch가 끝나면 손실함수와 정확도를 출력한다.
        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}'. \
                  format(epoch + 1, np.mean(losses), np.mean(accs), acc))
    #최종평가함수 run_test를 실행시킨 후, 결과를 출력한다.
    final_acc = run_test(test_x, test_y)
    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))

def arrange_data(mb_size):
    '''미니배치로 자르는 step을 계산한 후, return'''
    global data, shuffle_map, test_begin_idx
    # 데이터 수만큼 일련번호를 발생시킨 후 섞음.
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    #mini batch 처리 step의 수를 계산함.
    step_count = int(data.shape[0] * 0.8) // mb_size
    #split boundary training data and test data
    test_begin_idx = step_count * mb_size
    return step_count

def get_test_data():
    '''학습데이터와 정답데이터 분리'''
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]

def get_train_data(mb_size, nth):
    '''입력벡터와 정답벡터 분리'''
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:
        #처음 돌 때, training data에 대한 부분적인 순서를 섞어 epoch마다 다른 순서로 학습되게끔.
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size*nth:mb_size*(nth+1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]


def run_train(x, y):
    '''
    미니배치의 학습처리를 하는 함수
    aux_nn, aux_pp : backward propagation에서 쓰일 값을 forward propagation 과정에서 계산한 값을 저장해둠.
    1. forward , 2. accuracy계산 , 3. backward
    '''
    #입력값으로부터 출력값을 구하고 loss의 값도 구함.
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)

    accuracy = eval_accuracy(output, y)

    #aL/aL은 당연히 1
    G_loss = 1.0

    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)

    return loss, accuracy


def run_test(x, y):
    '''forward를 거친 후 정확도를 계산함.'''
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x


def backprop_neuralnet(G_output, x):
    '''실제 신경망의 하이퍼파라미터를 갱신해줌.'''
    global weight, bias
    g_output_w = x.transpose()

    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b


def forward_postproc(output, y):
    '''손실함수가 MSE니까 거기에 맞는 값을 계산한 후, return'''
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff


def backprop_postproc(G_loss, diff):
    '''

    return 2*diff /np.prod(diff.shpph) 로 줄일 수 있지만 과정을 보여주기 위해서 늘여씀.
    '''
    shape = diff.shape

    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff

    return G_output

def eval_accuracy(output, y):
    '''오차를 평균낸 후, 1에서 오차를 빼서 정확도를 구함'''
    mdiff = np.mean(np.abs((output - y)/y))
    return 1 - mdiff

#main part
'''
abalone_exec()
print(weight)
print(bias)
LEARNING_RATE = 0.1
abalone_exec(epoch_count=100, mb_size=100, report=20)
'''

