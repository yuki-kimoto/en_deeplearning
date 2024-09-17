use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/network_lib";
use List::Util 'shuffle';

use SPVM 'MyAIUtil2';
use SPVM 'Hash';
use SPVM 'List';
use SPVM 'IntList';

my $api = SPVM::api();

# Learning Rate
my $learning_rate = 3;

# Epoch count - the number of runs of the training set
my $epoch_count = 30;

# Mini-batch size
my $mini_batch_size = 10;

# Number of neurons in each layer
# 28 * 28 = 728 monochrome images         (input)
# Through 30 intermediate outputs         (Intermediate Output 1)
# Classify into 10 categories from 0 to 9 (output)
my $neurons_count_in_layers = $api->new_int_array([784, 30, 10]);

# Read MNIEST image information - handwritten training data to use as input
my $mnist_train_image_file = "$FindBin::Bin/data/train-images-idx3-ubyte";
my $mnist_train_image_info = load_mnist_train_image_file($mnist_train_image_file);

# Read MNIEST label information - expected output for handwritten training data
my $mnist_train_label_file = "$FindBin::Bin/data/train-labels-idx1-ubyte";
my $mnist_train_label_info = load_mnist_train_label_file($mnist_train_label_file);

# Converting MNIEST image information to SPVM data
my $mnist_train_image_info_spvm = SPVM::Hash->new($api->new_object_array('object[]', []));
$mnist_train_image_info_spvm->set_int(items_count => $mnist_train_image_info->{items_count});
$mnist_train_image_info_spvm->set_int(rows_count => $mnist_train_image_info->{rows_count});
$mnist_train_image_info_spvm->set_int(columns_count => $mnist_train_image_info->{columns_count});
$mnist_train_image_info_spvm->set(data => $api->new_byte_array_from_bin($mnist_train_image_info->{data}));

# Converting MNIEST label information to SPVM data
my $mnist_train_label_info_spvm = SPVM::Hash->new($api->new_object_array('object[]', []));
$mnist_train_label_info_spvm->set_int(items_count => $mnist_train_label_info->{items_count});
$mnist_train_label_info_spvm->set(label_numbers => SPVM::IntList->new($mnist_train_label_info->{label_numbers}));

# Train a deep network
SPVM::MyAIUtil2->train_deep_network($mnist_train_image_info_spvm, $mnist_train_label_info_spvm, $epoch_count, $mini_batch_size, $neurons_count_in_layers, $learning_rate);

# Load MNIST image information
sub load_mnist_train_image_file {
  my ($mnist_image_file) = @_;
  
  open my $mnist_image_fh, '<', $mnist_image_file
    or die "Can't open file $mnist_image_file: $!";

  # Magic Number
  my $image_buffer;
  read($mnist_image_fh, $image_buffer, 4);
  my $magic_number = unpack('N1', $image_buffer);
  if ($magic_number != 0x00000803) {
    die "Invalid magic number expected " . 0x00000803 . "actual $magic_number";
  }

  # Number of images
  read($mnist_image_fh, $image_buffer, 4);
  my $items_count = unpack('N1', $image_buffer);

  # Image row pixel count
  read($mnist_image_fh, $image_buffer, 4);
  my $rows_count = unpack('N1', $image_buffer);

  # Image column pixel count
  read($mnist_image_fh, $image_buffer, 4);
  my $columns_count = unpack('N1', $image_buffer);

  # Loading images
  my $image_data;
  my $all_images_length = $items_count * $rows_count * $columns_count;
  my $read_length = read $mnist_image_fh, $image_data, $all_images_length;
  unless ($read_length == $all_images_length) {
    die "Can't read all images";
  }

  # Image information
  my $image_info = {};
  $image_info->{items_count} = $items_count;
  $image_info->{rows_count} = $rows_count;
  $image_info->{columns_count} = $columns_count;
  $image_info->{data} = $image_data;
  
  return $image_info;
}

# Load MNIST image information
sub load_mnist_train_label_file {
  my ($mnist_label_file) = @_;

  open my $mnist_label_fh, '<', $mnist_label_file
    or die "Can't open file $mnist_label_file: $!";

  # Magic Number
  my $label_buffer;
  read($mnist_label_fh, $label_buffer, 4);
  my $magic_number = unpack('N1', $label_buffer);
  if ($magic_number != 0x00000801) {
    die "Invalid magic number expected " . 0x00000801 . "actual $magic_number";
  }

  # Number of labels
  read($mnist_label_fh, $label_buffer, 4);
  my $items_count = unpack('N1', $label_buffer);

  # Loading labels
  my $label_numbers = [];
  for (my $i = 0; $i < $items_count; $i++) {
    read $mnist_label_fh, $label_buffer, 1;
    my $label_number = unpack('C1', $label_buffer);
    push @$label_numbers, $label_number;
  }

  # Label Information
  my $label_info = {};
  $label_info->{items_count} = $items_count;
  $label_info->{label_numbers} = $label_numbers;
  
  return $label_info;
}

sub dump_array {
  my ($nums) = @_;
  
  my $dump_str = join(' ', @{$nums->to_elems});
  
  print STDERR "$dump_str\n";
}

sub dump_mat {
  my ($mat) = @_;
  
  my $values_str = join(' ', @{$mat->values->to_elems});
  my $dump_str = sprintf("rows_legnth: %d, columns_length: %d, values : %s", $mat->rows_length, $mat->columns_length, $values_str);
  
  print STDERR "$dump_str\n";
}
