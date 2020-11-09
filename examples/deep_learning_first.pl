=pod

このプログラムは

ニューラルネットワークと深層学習

https://nnadl-ja.github.io/nnadl_site_ja/

を参考にした、ディープラーニングプログラムのPerlへの最初の移植です。

参考にしたpythonプログラムは「python_examples/network.py」と「python_examples/network2.py」に置いてあります。


MNISTの手書き文字をディープラーニングを使って学習します。

速度は遅いですが、ライブラリも何も必要がなくピュアPerlで動作します。

ライブラリを使わずピュアPerlで実装したものとして世界初ではないかと思います。

if文とfor文の知識があれば、ディープラーニングのアルゴリズムのすべてを、読み解くことができます(がんばれば!!!)。

集合演算ライブラリがなければ、ディープラーニングを実装できないという誤解を解くプログラムともなっています。

速度向上は、SPVMと呼ばれるC言語トランスパイラを使うことで可能になります。

SPVMはPerl文法をサポートしたC言語トランスパイラで、いくらかのSPVMへの書き換えによって、速度向上を達成できます。

SPVMは、連続した数値の配列をサポートしているので、ディープラーニングに必要なfloat型の配列を表現できます。

パラメータ更新アルゴリズムは、バッチサイズを考慮した勾配降下法。

活性化関数は、シグモイド関数。

損失関数は、クロスエントロピーコスト。

正答率は、実行し続ければ、96%に到達(するはず)。

データは、THE MNIST DATABASE of handwritten digitsから取得して、dataディレクトリ以下に配置します。

http://yann.lecun.com/exdb/mnist/

画像の中身を出力したい場合は

parse_minist_image.pl

を使ってください。Imagerというモジュールが必要です。

=cut

use strict;
use warnings;
use FindBin;
use List::Util 'shuffle';

# 学習率
my $learning_rate = 3;

# エポック数 - 訓練セットの実行回数
my $epoch_count = 30;

# ミニバッチサイズ
my $mini_batch_size = 10;

# 各層のニューロンの数
# 28 * 28 = 784のモノクロ画像を (入力)
# 30個の中間出力を通って        (中間出力)
# 0～9の10個に分類する          (出力)
my $neurons_count_in_layers = [784, 30, 10];

# 各層のm個の入力をn個の出力に変換する関数の情報。入力数、出力数、バイアス、重み
my $m_to_n_func_infos = [];

# 各層のニューロン数からmからnへの変換関数の情報を作成
for (my $i = 0; $i < @$neurons_count_in_layers - 1; $i++) {
  my $inputs_length = $neurons_count_in_layers->[$i];
  my $outputs_length = $neurons_count_in_layers->[$i + 1];
  
  # バイアスを0で初期化
  my $biases = array_new_zero($outputs_length);
  
  # Xivierの初期値で重みを初期化。重みは列優先行列
  my $weights_mat = mat_new_zero($outputs_length, $inputs_length);
  my $weights_length = $weights_mat->{rows_length} * $weights_mat->{columns_length};
  $weights_mat->{values} = array_create_xavier_init_value($weights_length, $inputs_length);

  # 変換関数の情報を設定
  $m_to_n_func_infos->[$i] = {
    inputs_length => $inputs_length,
    outputs_length => $outputs_length,
    biases => $biases,
    weights_mat => $weights_mat,
  };
}

# MNIEST画像情報を読み込む - 入力用につかう手書きの訓練データ
my $mnist_train_image_file = "$FindBin::Bin/data/train-images-idx3-ubyte";
my $mnist_train_image_info = load_mnist_train_image_file($mnist_train_image_file);

# MNIESTラベル情報を読み込む - 手書きの訓練データの期待される出力
my $mnist_train_label_file = "$FindBin::Bin/data/train-labels-idx1-ubyte";
my $mnist_train_label_info = load_mnist_train_label_file($mnist_train_label_file);

# 訓練データのインデックス(最初の4万枚だけを訓練用データとして利用する。残りの1万枚は検証用データとする)
my @training_data_indexes = (0 .. 40000);

# ミニバッチ単位における各変換関数の情報
my $m_to_n_func_mini_batch_infos = [];

# ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
# メモリ領域を繰り返しつかうためここで初期化
for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_infos; $m_to_n_func_index++) {
  my $m_to_n_func_info = $m_to_n_func_infos->[$m_to_n_func_index];
  my $biases = $m_to_n_func_info->{biases};
  my $weights_mat = $m_to_n_func_info->{weights_mat};
  
  # バイアスの長さ
  my $biases_length = @$biases;
  
  # ミニバッチにおける各変換関数のバイアスの傾きの合計を0で初期化して作成
  my $biase_grad_totals = array_new_zero($biases_length);
  $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals} = $biase_grad_totals;

  # ミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
  my $weight_grad_totals_mat = mat_new_zero($weights_mat->{rows_length}, $weights_mat->{columns_length});
  $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat} = $weight_grad_totals_mat;
}

# 総実行回数
my $total_count = 0;

# 正解数
my $answer_match_count = 0;

# エポックの回数だけ訓練セットを実行
for (my $epoch_index = 0; $epoch_index < $epoch_count; $epoch_index++) {
  
  # 訓練データのインデックスをシャッフル(ランダムに学習させた方が汎用化するらしい)
  my @training_data_indexes_shuffle = shuffle @training_data_indexes;
  
  my $count = 0;
  
  # ミニバッチサイズ単位で学習
  my $backprop_count = 0;

  while (my @indexed_for_mini_batch = splice(@training_data_indexes_shuffle, 0, $mini_batch_size)) {
    
    # ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きの合計を0で初期化
    for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_mini_batch_infos; $m_to_n_func_index++) {
      my $m_to_n_func_info = $m_to_n_func_infos->[$m_to_n_func_index];
      my $biases = $m_to_n_func_info->{biases};
      my $weights_mat = $m_to_n_func_info->{weights_mat};
      
      # ミニバッチにおける各変換関数のバイアスの傾きの合計を0で初期化して作成
      $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals} = array_new_zero(scalar @{$m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals}});

      # ミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
      $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat}{values} = array_new_zero(scalar @{$m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat}{values}});
    }
    
    for my $training_data_index (@indexed_for_mini_batch) {
      # 逆誤差伝播法を使って重みとバイアスの損失関数に対する傾きを取得
      my $m_to_n_func_grad_infos = calculate_biase_and_weight_grads_to_cost_func_by_backprop($m_to_n_func_infos, $mnist_train_image_info, $mnist_train_label_info, $training_data_index);
      
      # バイアスの損失関数に対する傾き
      my $biase_grads = $m_to_n_func_grad_infos->{biases};
      
      # 重みの損失関数に対する傾き
      my $weight_grads_mat = $m_to_n_func_grad_infos->{weights_mat};

      # ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きを加算
      for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_mini_batch_infos; $m_to_n_func_index++) {
        my $m_to_n_func_info = $m_to_n_func_infos->[$m_to_n_func_index];
        
        # ミニバッチにおける各変換関数のバイアスの傾きを加算
        array_add_inplace($m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals}, $biase_grads->[$m_to_n_func_index]);

        # ミニバッチにおける各変換関数の重みの傾きを加算
        array_add_inplace($m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat}{values}, $weight_grads_mat->[$m_to_n_func_index]{values});
      }
    }

    # 各変換関数のバイアスと重みをミニバッチの傾きの合計を使って更新
    update_params($m_to_n_func_infos, $m_to_n_func_mini_batch_infos, $learning_rate, $mini_batch_size);
  }
}

# 学習率とミニバッチ数を考慮してパラメーターを更新
sub update_params {
  my ($m_to_n_func_infos, $m_to_n_func_mini_batch_infos, $learning_rate, $mini_batch_size) = @_;

  for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_infos; $m_to_n_func_index++) {
    
    # 各変換関数のバイアスを更新(学習率を考慮し、ミニバッチ数で割る)
    my $biases = $m_to_n_func_infos->[$m_to_n_func_index]{biases};
    my $biase_grad_totals = $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals};
    my $weights_mat_values = $m_to_n_func_infos->[$m_to_n_func_index]{weights_mat}{values};
    my $weight_grad_totals_mat_values = $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat}{values};
    
    # バイアスの更新 - パラメータ更新アルゴリズムにはAdamを使用
    for (my $biase_index = 0; $biase_index < @$biases; $biase_index++) {
      my $update_diff = ($learning_rate / $mini_batch_size) * $biase_grad_totals->[$biase_index];
      $biases->[$biase_index] -= $update_diff;
    }
    
    # 重みの更新 - パラメータ更新アルゴリズムにはAdamを使用
    for (my $weights_mat_values_index = 0; $weights_mat_values_index < @$weights_mat_values; $weights_mat_values_index++) {
      my $update_diff = ($learning_rate / $mini_batch_size) * $weight_grad_totals_mat_values->[$weights_mat_values_index];
      $weights_mat_values->[$weights_mat_values_index] -= $update_diff;
    }
  }
}

my $count = 0;

# 逆誤差伝播法を使ってすべての重みとバイアスの損失関数に対する傾きを求める
sub calculate_biase_and_weight_grads_to_cost_func_by_backprop {
  my ($m_to_n_func_infos, $mnist_train_image_info, $mnist_train_label_info, $training_data_index) = @_;
  
  my $first_inputs_length = $m_to_n_func_infos->[0]{inputs_length};
  
  # 入力(0～255の値を0～1に変換)
  my $image_unit_length = $mnist_train_image_info->{rows_count} *  $mnist_train_image_info->{columns_count};
  my $mnist_train_image_data = $mnist_train_image_info->{data};
  my $first_inputs_packed = substr($mnist_train_image_data, $image_unit_length * $training_data_index, $image_unit_length);
  my $first_inputs_raw = [unpack("C$first_inputs_length", $first_inputs_packed)];
  my $first_inputs = array_div_scalar($first_inputs_raw, 255);
  
  # 期待される出力を確率分布化する
  my $label_number = $mnist_train_label_info->{label_numbers}[$training_data_index];
  my $desired_outputs = probabilize_desired_outputs($label_number);
  
  # 各変換関数のバイアスの傾き
  my $biase_grads_in_m_to_n_funcs = [];
  
  # 各変換関数の重みの傾き
  my $weight_grads_mat_in_m_to_n_funcs = [];
  
  # バイアスの傾きと重みの傾きの初期化
  for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_infos; $m_to_n_func_index++) {
    my $inputs_length = $m_to_n_func_infos->[$m_to_n_func_index]{inputs_length};
    my $outputs_length = $m_to_n_func_infos->[$m_to_n_func_index]{outputs_length};

    # バイアスの傾きを0で初期化
    $biase_grads_in_m_to_n_funcs->[$m_to_n_func_index] = array_new_zero($outputs_length);

    # 重みの傾きを0で初期化
    $weight_grads_mat_in_m_to_n_funcs->[$m_to_n_func_index] = mat_new_zero($outputs_length, $inputs_length);
  }

  # 各層の入力
  my $inputs_in_m_to_n_funcs = [$first_inputs];
  
  # 各層の活性化された出力
  my $outputs_in_m_to_n_funcs = [];
  
  # 入力層の入力から出力層の出力を求める
  # 逆誤差伝播法のために各層の出力と活性化された出力を保存
  for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_infos; $m_to_n_func_index++) {
    my $cur_inputs = $inputs_in_m_to_n_funcs->[-1];
    my $inputs_length = $m_to_n_func_infos->[$m_to_n_func_index]{inputs_length};
    my $outputs_length = $m_to_n_func_infos->[$m_to_n_func_index]{outputs_length};
    
    # 重み行列
    my $weights_mat = $m_to_n_func_infos->[$m_to_n_func_index]{weights_mat};
    
    # 入力行列
    my $cur_inputs_rows_length = $outputs_length;
    my $cur_inputs_columns_length = 1;
    my $cur_inputs_mat = {
      rows_length => $cur_inputs_rows_length,
      columns_length => $cur_inputs_columns_length,
      values => $cur_inputs,
    };
    
    # 重みと入力の行列積
    my $mul_weights_inputs_mat = mat_mul($weights_mat, $cur_inputs_mat);
    my $mul_weights_inputs = $mul_weights_inputs_mat->{values};
    
    # バイアス
    my $biases = $m_to_n_func_infos->[$m_to_n_func_index]{biases};
    
    # 出力 - 重みと入力の行列積とバイアスの和
    my $outputs = array_add($mul_weights_inputs, $biases);
    
    # 活性化された出力 - 出力に活性化関数を適用
    my $activate_outputs = array_sigmoid($outputs);
    
    # 逆誤差伝播法のために出力を保存
    push @$outputs_in_m_to_n_funcs, $outputs;
    
    # 逆誤差伝播法のために次の入力を保存
    push @$inputs_in_m_to_n_funcs, $activate_outputs;
  }

  
  # 最後の出力
  my $last_outputs = $outputs_in_m_to_n_funcs->[-1];
  
  # 最後の活性化された出力(最後に追加されたものは次の入力にならないのでポップする)
  my $last_activate_outputs = pop @$inputs_in_m_to_n_funcs;
  
  # 誤差の指標となる損失関数
  my $cost = cross_entropy_cost($last_activate_outputs, $desired_outputs);
  print "Cost: " . sprintf("%.3f", $cost) . "\n";
  
  # 正解したかどうか
  my $answer = max_index($last_activate_outputs);
  my $desired_answer = max_index($desired_outputs);
  $total_count++;
  if ($answer == $desired_answer) {
    $answer_match_count++;
  }
  
  # 正解率を出力
  my $match_rate = $answer_match_count / $total_count;
  print "Match Rate: " . sprintf("%.02f", 100 * $match_rate) . "%\n";
  
  # 活性化された出力の微小変化 / 最後の出力の微小変化 
  my $grad_last_outputs_to_activate_func = array_sigmoid_derivative($last_outputs);
  
  # 損失関数の微小変化 / 最後に活性化された出力の微小変化
  my $grad_last_activate_outputs_to_cost_func = cross_entropy_cost_derivative($last_activate_outputs, $desired_outputs);

  # 損失関数の微小変化 / 最後の出力の微小変化 (合成微分)
  my $grad_last_outputs_to_cost_func = array_mul($grad_last_outputs_to_activate_func, $grad_last_activate_outputs_to_cost_func);

  # 損失関数の微小変化 / 最終の層のバイアスの微小変化
  my $last_biase_grads = $grad_last_outputs_to_cost_func;
  
  # 損失関数の微小変化 / 最終の層の重みの微小変化
  my $last_biase_grads_mat = array_to_mat($last_biase_grads);
  my $last_inputs = $inputs_in_m_to_n_funcs->[@$inputs_in_m_to_n_funcs - 1];
  my $last_inputs_mat_transposed = array_to_mat_transposed($last_inputs);
  my $last_weight_grads_mat = mat_mul($last_biase_grads_mat, $last_inputs_mat_transposed);
  
  $biase_grads_in_m_to_n_funcs->[@$biase_grads_in_m_to_n_funcs - 1] = $last_biase_grads;
  $weight_grads_mat_in_m_to_n_funcs->[@$biase_grads_in_m_to_n_funcs - 1] = $last_weight_grads_mat;

  # 最後の重みとバイアスの変換より一つ前から始める
  for (my $m_to_n_func_index = @$m_to_n_func_infos - 2; $m_to_n_func_index >= 0; $m_to_n_func_index--) {
    # 活性化された出力の微小変化 / 出力の微小変化
    my $outputs = $outputs_in_m_to_n_funcs->[$m_to_n_func_index];

    # 損失関数の微小変化 / この層のバイアスの微小変化(逆誤差伝播法で求める)
    # 次の層の重みの転置行列とバイアスの傾きをかけて、それぞれの要素に、活性化関数の導関数をかける
    my $next_weights_mat = $m_to_n_func_infos->[$m_to_n_func_index + 1]{weights_mat};
    my $next_weights_mat_transposed = mat_transpose($next_weights_mat);
    my $next_biase_grads = $biase_grads_in_m_to_n_funcs->[$m_to_n_func_index + 1];
    my $next_biase_grads_mat = array_to_mat($next_biase_grads);
    my $mul_forword_weights_transpose_mat_forword_biase_grads_mat = mat_mul($next_weights_mat_transposed, $next_biase_grads_mat);
    my $mul_forword_weights_transpose_mat_forword_biase_grads_mat_values = $mul_forword_weights_transpose_mat_forword_biase_grads_mat->{values};
    my $grads_outputs_to_array_sigmoid = array_sigmoid_derivative($outputs);
    my $biase_grads = array_mul($mul_forword_weights_transpose_mat_forword_biase_grads_mat_values, $grads_outputs_to_array_sigmoid);

    $biase_grads_in_m_to_n_funcs->[$m_to_n_func_index] = $biase_grads;
    
    # 損失関数の微小変化 / この層の重みの微小変化(逆誤差伝播法で求める)
    my $biase_grads_mat = array_to_mat($biase_grads);
    my $inputs = $inputs_in_m_to_n_funcs->[$m_to_n_func_index];
    my $inputs_mat_transposed = array_to_mat_transposed($inputs);
    my $weights_grads_mat = mat_mul($biase_grads_mat, $inputs_mat_transposed);
    
    $weight_grads_mat_in_m_to_n_funcs->[$m_to_n_func_index] = $weights_grads_mat;
  }

  my $m_to_n_func_grad_infos = {};
  $m_to_n_func_grad_infos->{biases} = $biase_grads_in_m_to_n_funcs;
  $m_to_n_func_grad_infos->{weights_mat} = $weight_grads_mat_in_m_to_n_funcs;
  
  return $m_to_n_func_grad_infos;
}

# 配列の中で最大値のインデックスを求める。同じ数の場合は、最初の方を返す
sub max_index {
  my ($nums) = @_;
  
  my $max = $nums->[0];
  my $max_index = 0;
  for (my $i = 0; $i < @$nums; $i++) {
    if ($nums->[$i] > $max) {
      $max_index = $i;
      $max = $nums->[$i];
    }
  }
  
  return $max_index;
}

# 期待される出力を確率分布化する
sub probabilize_desired_outputs {
  my ($label_number) = @_;
  
  my $desired_outputs = [];
  for (my $desired_outputs_index = 0; $desired_outputs_index < 10; $desired_outputs_index++) {
    if ($label_number == $desired_outputs_index) {
      $desired_outputs->[$desired_outputs_index] = 1;
    }
    else {
      $desired_outputs->[$desired_outputs_index] = 0;
    }
  }
  
  return $desired_outputs;
}

# 配列の各要素の和
sub array_add {
  my ($nums1, $nums2) = @_;
  
  if (@$nums1 != @$nums2) {
    die "Array length is diffent";
  }
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums1; $i++) {
    $nums_out->[$i] = $nums1->[$i] + $nums2->[$i];
  }
  
  return $nums_out;
}

# 配列の各要素の和を最初の引数に足す
sub array_add_inplace {
  my ($nums1, $nums2) = @_;
  
  if (@$nums1 != @$nums2) {
    die "Array length is diffent";
  }
  
  for (my $i = 0; $i < @$nums1; $i++) {
    $nums1->[$i] += $nums2->[$i];
  }
}

# 配列の各要素の積
sub array_mul {
  my ($nums1, $nums2) = @_;
  
  if (@$nums1 != @$nums2) {
    die "Array length is diffent";
  }
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums1; $i++) {
    $nums_out->[$i] = $nums1->[$i] * $nums2->[$i];
  }
  
  return $nums_out;
}

# 正規分布に従う乱数を求める関数
# $m は平均, $sigma は標準偏差、
sub randn {
  my ($m, $sigma) = @_;
  my ($r1, $r2) = (rand(), rand());
  while ($r1 == 0) { $r1 = rand(); }
  return ($sigma * sqrt(-2 * log($r1)) * sin(2 * 3.14159265359 * $r2)) + $m;
}

# Xivierの初期値を作成
sub create_xavier_init_value {
  my ($inputs_length) = @_;
  
  return randn(0, sqrt(1 / $inputs_length));
}

# 配列の各要素にXivierの初期値を取得を適用する
sub array_create_xavier_init_value {
  my ($array_length, $inputs_length) = @_;
  
  my $nums_out = [];
  for (my $i = 0; $i < $array_length; $i++) {
    $nums_out->[$i] = create_xavier_init_value($inputs_length);
  }
  
  return $nums_out;
}

# シグモイド関数
sub sigmoid {
  my ($x) = @_;
  
  my $sigmoid = 1.0 / (1.0 + exp(-$x));
  
  return $sigmoid;
}

# シグモイド関数の導関数
sub sigmoid_derivative {
  my ($x) = @_;
  
  my $sigmoid_derivative = sigmoid($x) * (1 - sigmoid($x));
  
  return $sigmoid_derivative;
}

# 配列の各要素にシグモイド関数を適用する
sub array_sigmoid {
  my ($nums) = @_;
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums; $i++) {
    $nums_out->[$i] = sigmoid($nums->[$i]);
  }
  
  return $nums_out;
}

# 配列の各要素にシグモイド関数の導関数を適用する
sub array_sigmoid_derivative {
  my ($nums) = @_;
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums; $i++) {
    $nums_out->[$i] = sigmoid_derivative($nums->[$i]);
  }
  
  return $nums_out;
}

# クロスエントロピーコスト
sub cross_entropy_cost {
  my ($vec_a, $vec_y) = @_;
  
  my $cross_entropy_cost = 0;
  for (my $i = 0; $i < @$vec_a; $i++) {
    my $tmp = -$vec_y->[$i] * log($vec_a->[$i]) - (1 - $vec_y->[$i]) * log(1 - $vec_a->[$i]);
    $cross_entropy_cost += $tmp;
  }

  return $cross_entropy_cost;
}

# クロスエントロピーコストの導関数
sub cross_entropy_cost_derivative {
  my ($vec_a, $vec_y) = @_;
  
  my $vec_out = [];
  for (my $i = 0; $i < @$vec_a; $i++) {
    $vec_out->[$i] = $vec_a->[$i] - $vec_y->[$i];
  }
  
  return $vec_out;
}

# MNIST画像情報を読み込む
sub load_mnist_train_image_file {
  my ($mnist_image_file) = @_;
  
  open my $mnist_image_fh, '<', $mnist_image_file
    or die "Can't open file $mnist_image_file: $!";

  # マジックナンバー
  my $image_buffer;
  read($mnist_image_fh, $image_buffer, 4);
  my $magic_number = unpack('N1', $image_buffer);
  if ($magic_number != 0x00000803) {
    die "Invalid magic number expected " . 0x00000803 . "actual $magic_number";
  }

  # 画像数
  read($mnist_image_fh, $image_buffer, 4);
  my $items_count = unpack('N1', $image_buffer);

  # 画像の行ピクセル数
  read($mnist_image_fh, $image_buffer, 4);
  my $rows_count = unpack('N1', $image_buffer);

  # 画像の列ピクセル数
  read($mnist_image_fh, $image_buffer, 4);
  my $columns_count = unpack('N1', $image_buffer);

  # 画像の読み込み
  my $image_data;
  my $all_images_length = $items_count * $rows_count * $columns_count;
  my $read_length = read $mnist_image_fh, $image_data, $all_images_length;
  unless ($read_length == $all_images_length) {
    die "Can't read all images";
  }

  # 画像情報
  my $image_info = {};
  $image_info->{items_count} = $items_count;
  $image_info->{rows_count} = $rows_count;
  $image_info->{columns_count} = $columns_count;
  $image_info->{data} = $image_data;
  
  return $image_info;
}

# MNIST画像情報を読み込む
sub load_mnist_train_label_file {
  my ($mnist_label_file) = @_;

  open my $mnist_label_fh, '<', $mnist_label_file
    or die "Can't open file $mnist_label_file: $!";

  # マジックナンバー
  my $label_buffer;
  read($mnist_label_fh, $label_buffer, 4);
  my $magic_number = unpack('N1', $label_buffer);
  if ($magic_number != 0x00000801) {
    die "Invalid magic number expected " . 0x00000801 . "actual $magic_number";
  }

  # ラベル数
  read($mnist_label_fh, $label_buffer, 4);
  my $items_count = unpack('N1', $label_buffer);

  # ラベルの読み込み
  my $label_numbers = [];
  for (my $i = 0; $i < $items_count; $i++) {
    read $mnist_label_fh, $label_buffer, 1;
    my $label_number = unpack('C1', $label_buffer);
    push @$label_numbers, $label_number;
  }

  # ラベル情報
  my $label_info = {};
  $label_info->{items_count} = $items_count;
  $label_info->{label_numbers} = $label_numbers;
  
  return $label_info;
}

# 配列を0で初期化して作成
sub array_new_zero {
  my ($length) = @_;
  
  my $nums = [(0) x $length];
  
  return $nums;
}

# 行列を0で初期化
sub mat_new_zero {
  my ($rows_length, $columns_length) = @_;
  
  my $values_length = $rows_length * $columns_length;
  my $mat = {
    rows_length => $rows_length,
    columns_length => $columns_length,
    values => [(0) x $values_length],
  };
  
  return $mat;
}

# 行列の積を求める
sub mat_mul {
  my ($mat1, $mat2) = @_;
  
  my $mat1_rows_length = $mat1->{rows_length};
  my $mat1_columns_length = $mat1->{columns_length};
  my $mat1_values = $mat1->{values};
  
  my $mat2_rows_length = $mat2->{rows_length};
  my $mat2_columns_length = $mat2->{columns_length};
  my $mat2_values = $mat2->{values};
  
  # 行列の積の計算
  my $mat_out_values = [];
  for(my $row = 0; $row < $mat1_rows_length; $row++) {
    for(my $col = 0; $col < $mat2_columns_length; $col++) {
      for(my $incol = 0; $incol < $mat1_columns_length; $incol++) {
        $mat_out_values->[$row + $col * $mat1_rows_length]
         += $mat1_values->[$row + $incol * $mat1_rows_length] * $mat2_values->[$incol + $col * $mat2_rows_length];
      }
    }
  }
  
  my $mat_out = {
    rows_length => $mat1_rows_length,
    columns_length => $mat2_columns_length,
    values => $mat_out_values,
  };
  
  return $mat_out;
}

# 列優先の行列の作成
sub mat_new {
  my ($values, $rows_length, $columns_length) = @_;
  
  my $mat = {
    rows_length => $rows_length,
    columns_length => $columns_length,
    values => $values,
  };
  
  return $mat;
}

# 行列を転置(行列の入れ替え)
sub mat_transpose {
  my ($mat) = @_;
  
  my $rows_length = $mat->{rows_length};
  my $columns_length = $mat->{columns_length};
  my $length = $rows_length * $columns_length;
  
  my $mat_trans ={};
  $mat_trans->{rows_length} = $columns_length;
  $mat_trans->{columns_length} = $rows_length;
  
  my $values = $mat->{values};
  my $mat_trans_values = [];
  
  for (my $row_index = 0; $row_index < $rows_length; $row_index++) {
    for (my $column_index = 0; $column_index < $columns_length; $column_index++) {
      $mat_trans_values->[$row_index * $columns_length + $column_index] = $values->[$column_index * $rows_length+ $row_index];
    }
  }
  $mat_trans->{values} = $mat_trans_values;
  
  return $mat_trans;
}

# 配列の値を、一つの値で除算
sub array_div_scalar {
  my ($nums, $scalar_num) = @_;
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums; $i++) {
    $nums_out->[$i] = $nums->[$i] / $scalar_num;
  }
  
  return $nums_out;
}

# 配列を行列に変換
sub array_to_mat {
  my ($values) = @_;
  
  my $rows_length = @$values;
  
  my $mat = {
    rows_length => $rows_length,
    columns_length => 1,
    values => $values,
  };
  
  return $mat;
}

# 配列を転置された行列に変換
sub array_to_mat_transposed {
  my ($values) = @_;
  
  my $rows_length = @$values;
  
  my $mat = {
    rows_length => 1,
    columns_length => $rows_length,
    values => $values,
  };
  
  return $mat;
}
