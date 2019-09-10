from utils import *
from model import *
from solver import *

SEED = 2019
path = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
output_path = './'
EMBEDDING_FILES = [
    # '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl',
    # '../input/pickled-paragram-300-vectors-sl999/paragram_300_sl999.pkl',
    '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
]

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=220)
parser.add_argument('--vocab-size', type=int, default=100000)
parser.add_argument('--n-splits', type=int, default=5,
                    help='splits of n-fold cross validation')
parser.add_argument('--nb-models', type=int, default=5,
                    help='number of models (folds) to ensemble')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--lr', type=float, default=5e-3)
args = parser.parse_args()


def train_val_split(train_x, train_y, nb_models=args.nb_models):
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in kf.split(train_x, train_y)]
    if nb_models:
        return cv_indices[:nb_models]
    return cv_indices

def convert_dataframe_to_bool(df):
    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    bool_df = df.copy()
    for col in [label_column] + identity_columns + aux_columns:
        convert_to_bool(bool_df, col)
    return bool_df

def load_and_preproc():
    train_df = pd.read_csv(path+'train.csv')
    test_df = pd.read_csv(path+'test.csv')
    train_df[identity_columns] = train_df[identity_columns].copy().fillna(0)

    print('cleaning text...')
    t0 = time.time()
    train_df[text_column] = train_df[text_column].apply(clean_text)
    test_df[text_column] = test_df[text_column].apply(clean_text)
    print('cleaning complete in {:.0f} seconds.'.format(time.time()-t0))

    sample_weights = np.ones(len(train_df))
    sample_weights += train_df[identity_columns].values.sum(1) * 3
    sample_weights += train_df[label_column].values * 8
    sample_weights /= sample_weights.max()
    train_tars = train_df[label_column].values
    train_tars = np.vstack([train_tars, sample_weights]).T.astype('float32')

    train_df = convert_dataframe_to_bool(train_df)
    df = train_df[[label_column]+identity_columns].copy()
    df[label_column] = df[label_column].astype('uint8')

    return train_df[text_column], train_tars, test_df[text_column], df, test_df['id']



## main()
np.random.seed(SEED)
train_seq, train_tars, x_test, trn_df, test_id = load_and_preproc()

print('tokenizing...')
t0 = time.time()
word_to_idx, idx_to_word = word_idx_map(train_seq.tolist()+x_test.tolist(), args.vocab_size)
train_seq = tokenize(train_seq, word_to_idx, args.maxlen)
print('tokenizing complete in {:.0f} seconds.'.format(time.time()-t0))


print('loading embeddings...')
t0 = time.time()
embed_mat = np.concatenate(
    [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=-1)
# embed_mat = np.mean(
#     [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=0)
print('loading complete in {:.0f} seconds.'.format(time.time()-t0))


# preparation
cv_indices = train_val_split(train_seq, (train_tars[:,0]>=0.5).astype(int))

print()
for i, (trn_idx, val_idx) in enumerate(cv_indices):
    print(f'Fold {i + 1}')

    # train/val split
    x_train, x_val = train_seq[trn_idx], train_seq[val_idx]
    y_train, y_val = train_tars[trn_idx], train_tars[val_idx]
    train_loader = prepare_loader(x_train, y_train, args.batch_size, split='train')

    # model setup
    seed_torch(SEED+i)
    model = JigsawNet(*embed_mat.shape, 128, embed_mat)
    for name, param in model.named_parameters():
      if 'emb' in name:
          param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # ft_lrs = [args.lr*0.08, args.lr, args.lr]
    # model, optimizer = model_optimizer_init(160, embed_mat, ft_lrs)
    model = model.to(device)
    criterion = UnbiasLoss().to(device)

    # lr range test
    lrs_logs, loss_log = lr_range_test(train_loader, model, optimizer, criterion)
    lrs_logs = list(zip(*lrs_logs))
    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(lrs_logs[0][10:-3], loss_log[10:-3])
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning rate')
    ax1.set_ylabel('Loss')
    # lrs_logs, loss_log = solver.lr_range_test(train_loader, start_lr=(1e-7*0.08,1e-7,1e-7), end_lr=(0.8,10,10))
    # lrs_logs = list(zip(*lrs_logs))
    # fig = plt.figure(figsize=(8,5))
    # ax1 = fig.add_subplot(1,2,1)
    # ax1.plot(lrs_logs[0][10:-3], loss_log[10:-3])
    # ax1.set_xscale('log')
    # ax1.set_xlabel('Learning rate')
    # ax1.set_ylabel('Loss')
    # ax2 = fig.add_subplot(1,2,2)
    # ax2.plot(lrs_logs[1][10:-3], loss_log[10:-3])
    # ax2.set_xscale('log')
    # ax2.set_xlabel('Learning rate')
    # ax2.set_ylabel('Loss')
    fig.savefig(f'lr_range_{i+1}.png')

    print()


