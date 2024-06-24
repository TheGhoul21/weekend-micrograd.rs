use rand::Rng;

pub mod param {
    use std::{
        collections::HashSet,
        ops::{Add, AddAssign, Div, Mul, Neg, Sub},
        sync::Mutex,
        vec,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Op {
        Add,
        Mul,
        Init,
        Relu,
        Exp,
        Pow,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Param {
        id: u64,
    }

    impl Param {
        // fn id(&self) -> u64 {
        //     self.id
        // }

        fn prev(&self) -> Vec<Param> {
            let pm = PARAM_MANAGER.lock().unwrap();

            let param = pm.get_param(self).unwrap();

            param.prev.clone()
        }

        pub fn relu(&self) -> Param {
            ParamManager::relu(self)
        }

        pub fn exp(&self) -> Param {
            ParamManager::exp(self)
        }

        pub fn set_value(&self, value: f32) {
            ParamManager::set_value(self, value);
        }

        pub fn powf(&self, power: f32) -> Param {
            ParamManager::powf(self, power)
        }

        pub fn grad(&self) -> f32 {
            let pm = PARAM_MANAGER.lock().unwrap();

            let param = pm.get_param(self).unwrap();

            param.grad
        }

        pub fn set_grad(&self, grad: f32) {
            let mut pm = PARAM_MANAGER.lock().unwrap();
            pm.set_grad(&self, grad);
        }

        pub fn value(&self) -> f32 {
            let pm = PARAM_MANAGER.lock().unwrap();

            let value = pm.get_value(self);
            value
        }

        pub fn op(&self) -> Op {
            let pm = PARAM_MANAGER.lock().unwrap();

            let param = pm.get_param(self).unwrap();

            param.op
        }

        pub fn _backward_add(&self) {
            let prev = self.prev();

            let id1 = prev[0];
            let id2 = prev[1];

            let grad = self.grad();

            id1.acc_grad(grad);
            id2.acc_grad(grad);
        }

        pub fn _backward_pow(&self) {
            let prev = self.prev();

            let id1 = prev[0];
            let id2 = prev[1];

            let grad = self.grad();

            id1.acc_grad(grad * id2.value() * id1.value().powf(id2.value() - 1.0));
        }

        pub fn _backward_mul(&self) {
            let prev = self.prev();

            let id1 = prev[0];
            let id2 = prev[1];

            let grad = self.grad();
            id1.acc_grad(grad * id2.value());
            id2.acc_grad(grad * id1.value());
        }

        pub fn _backward_relu(&self) {
            let prev = self.prev();

            let id1 = prev[0];

            if id1.value() > 0.0 {
                id1.acc_grad(self.grad());
            }
        }

        pub fn _backward_exp(&self) {
            let prev = self.prev();

            let id1 = prev[0];

            let grad = self.grad();

            id1.acc_grad(grad * self.value());
        }

        pub fn _backward(&self) {
            let op = self.op();

            match op {
                Op::Add => {
                    self._backward_add();
                }
                Op::Mul => {
                    self._backward_mul();
                }
                Op::Pow => {
                    self._backward_pow();
                }
                Op::Relu => {
                    self._backward_relu();
                }
                Op::Exp => {
                    self._backward_exp();
                }
                _ => {
                    // println!("Not implemented");
                }
            }
        }

        pub fn acc_grad(&self, grad: f32) {
            // self.grad = grad;
            // ParamManager::get_param(&self, id)
            let mut pm = PARAM_MANAGER.lock().unwrap();

            let ids = vec![self];

            pm.acc_grad_multiple(ids, grad);
            drop(pm);
        }

        pub fn backward(&mut self) {
            let mut topo: Vec<u64> = vec![];
            let mut visited: HashSet<u64> = HashSet::new();

            fn build_topo(
                node: &Param,
                topo: &mut Vec<u64>,
                visited: &mut HashSet<u64>,
            ) -> Vec<u64> {
                if !visited.contains(&node.id) {
                    visited.insert(node.id);
                    let prev = node.prev();
                    for prev_node in prev.iter() {
                        build_topo(prev_node, topo, visited);
                    }
                    topo.push(node.id);
                }

                topo.to_vec()
            }
            self.set_grad(1.0);
            let mut topo = build_topo(self, &mut topo, &mut visited);

            let drained = topo.drain(0..topo.len()).collect::<Vec<u64>>();
            let mut reversed = drained;
            reversed.reverse();

            let mut node = Param { id: self.id };
            for id in reversed.drain(..) {
                node.id = id;
                node._backward();
            }
        }
    }

    #[derive(Debug)]
    pub struct ParamContainer {
        id: Param,
        value: f32,
        prev: Vec<Param>,
        op: Op,
        pub grad: f32,
    }

    impl Drop for ParamContainer {
        fn drop(&mut self) {
            // println!("Dropping {:?}", self.id);
        }
    }

    impl ParamContainer {
        fn acc_grad(&mut self, grad: f32) {
            self.grad += grad;
        }
    }

    lazy_static::lazy_static! {
        pub static ref PARAM_MANAGER: Mutex<ParamManager> = Mutex::new(ParamManager::init());
    }

    pub struct ParamManager {
        params: Vec<ParamContainer>,
    }

    pub type PM = ParamManager;

    impl ParamManager {
        pub fn init() -> ParamManager {
            ParamManager { params: Vec::new() }
        }

        fn next_id(&self) -> u64 {
            self.params.len() as u64
        }

        fn set_value(id: &Param, value: f32) {
            let mut pm = PARAM_MANAGER.lock().unwrap();
            let param = pm.get_param_mut(id).unwrap();
            param.value = value;
        }

        pub fn get_next_id() -> u64 {
            let pm = PARAM_MANAGER.lock().unwrap();
            pm.next_id()
        }

        pub fn dump_all_params(&self) {
            for p in self.params.iter() {
                println!("{:?}", p);
            }
        }

        pub fn add_param(value: f32, prev: Vec<Param>, op: Op) -> Param {
            let mut pm = PARAM_MANAGER.lock().unwrap();
            let id = pm.next_id();
            let param = ParamContainer {
                id: Param { id },
                value,
                grad: 0.0,
                op,
                prev,
            };

            pm.params.push(param);
            Param { id }
        }

        pub fn from_value(value: f32) -> Param {
            ParamManager::add_param(value, Vec::new(), Op::Init)
        }
        pub fn powf(id: &Param, power: f32) -> Param {
            let value = id.value().powf(power);
            let prev = vec![*id, ParamManager::add_param(power, Vec::new(), Op::Init)];
            let op = Op::Pow;

            ParamManager::add_param(value, prev, op)
        }

        pub fn get_param(&self, id: &Param) -> Option<&ParamContainer> {
            self.params.iter().find(|p| (**p).id.id == id.id)
        }

        pub fn get_param_container(&self, id: &Param) -> Option<&ParamContainer> {
            self.params.iter().find(|p| (**p).id.id == id.id)
        }

        pub fn get_param_mut(&mut self, id: &Param) -> Option<&mut ParamContainer> {
            self.params.iter_mut().find(|p| (**p).id.id == id.id)
        }

        pub fn get_value(&self, id: &Param) -> f32 {
            match self.get_param(id) {
                Some(p) => p.value,
                None => {
                    println!("No param found");
                    0.0
                }
            }
        }
        pub fn get_container(&self, id: &Param) -> Option<&ParamContainer> {
            match self.get_param(id) {
                Some(p) => Some(p),
                None => {
                    println!("No param found");
                    None
                }
            }
        }

        pub fn relu(id: &Param) -> Param {
            let value = id.value().max(0.0);
            let prev = vec![*id];
            let op = Op::Relu;

            ParamManager::add_param(value, prev, op)
        }

        pub fn exp(id: &Param) -> Param {
            let value = id.value().exp();
            let prev = vec![*id];
            let op = Op::Exp;

            ParamManager::add_param(value, prev, op)
        }

        pub fn acc_grad(&mut self, id: &Param, grad: f32) {
            match self.get_param_mut(id) {
                Some(p) => {
                    p.acc_grad(grad);
                }
                None => {
                    println!("No param found");
                }
            }
        }

        pub fn acc_grad_multiple(&mut self, ids: Vec<&Param>, grad: f32) {
            for id in ids {
                self.acc_grad(id, grad);
            }
        }

        pub fn set_grad(&mut self, id: &Param, grad: f32) {
            match self.get_param_mut(id) {
                Some(p) => {
                    p.grad = grad;
                }
                None => {
                    println!("No param found");
                }
            }
        }
    }

    fn _add_fn(p1: Param, p2: Param) -> Param {
        let value = p1.value() + p2.value();
        let prev = vec![p1, p2];
        let op = Op::Add;

        ParamManager::add_param(value, prev, op)
    }

    impl Add for Param {
        type Output = Param;
        fn add(self, other: Param) -> Param {
            _add_fn(self, other)
        }
    }
    impl Add<f32> for Param {
        type Output = Param;

        fn add(self, rhs: f32) -> Param {
            _add_fn(self, ParamManager::from_value(rhs))
        }
    }

    impl Add<Param> for f32 {
        type Output = Param;

        fn add(self, rhs: Param) -> Param {
            _add_fn(ParamManager::from_value(self), rhs)
        }
    }

    impl Neg for Param {
        type Output = Param;
        fn neg(self) -> Self::Output {
            self * -1.0
        }
    }

    impl Div<Param> for f32 {
        type Output = Param;

        fn div(self, rhs: Param) -> Param {
            self * ParamManager::powf(&rhs, -1.0)
        }
    }
    impl Div<f32> for Param {
        type Output = Param;

        fn div(self, rhs: f32) -> Param {
            self * ParamManager::powf(&ParamManager::from_value(rhs), -1.0)
        }
    }

    impl Div for Param {
        type Output = Param;

        fn div(self, rhs: Param) -> Param {
            self * ParamManager::powf(&rhs, -1.0)
        }
    }

    fn _mul_fn(p1: Param, p2: Param) -> Param {
        let value = p1.value() * p2.value();
        let prev = vec![p1, p2];
        let op = Op::Mul;

        ParamManager::add_param(value, prev, op)
    }

    impl Mul for Param {
        type Output = Param;

        fn mul(self, other: Param) -> Param {
            _mul_fn(self, other)
        }
    }
    impl Mul<f32> for Param {
        type Output = Param;

        fn mul(self, rhs: f32) -> Param {
            _mul_fn(self, ParamManager::from_value(rhs))
        }
    }
    impl Mul<Param> for f32 {
        type Output = Param;

        fn mul(self, rhs: Param) -> Param {
            _mul_fn(ParamManager::from_value(self), rhs)
        }
    }

    impl AddAssign for Param {
        fn add_assign(&mut self, other: Param) {
            *self = _add_fn(*self, other);
        }
    }
    impl AddAssign<f32> for Param {
        fn add_assign(&mut self, rhs: f32) {
            *self = _add_fn(*self, ParamManager::from_value(rhs));
        }
    }

    impl Sub for Param {
        type Output = Param;

        fn sub(self, other: Param) -> Param {
            self + (-other)
        }
    }

    impl Sub<f32> for Param {
        type Output = Param;

        fn sub(self, rhs: f32) -> Param {
            self + (-ParamManager::from_value(rhs))
        }
    }

    impl Sub<Param> for f32 {
        type Output = Param;

        fn sub(self, rhs: Param) -> Param {
            ParamManager::from_value(self) + (-rhs)
        }
    }
}

use param::Param;

use crate::param::PM;
fn main() {
    // let id1 = PM::add_param(1.0, Vec::new(), Op::Init);
    // let id2 = PM::add_param(3.0, Vec::new(), Op::Init);
    // let id3 = PM::add_param(5.0, Vec::new(), Op::Init);
    // println!("{:?}", id1);
    // println!("{:?}", id2);
    // println!("{:?}", id3);

    // let id4 = id1 + id2;
    // let id5 = id4 * id3;

    // let id6 = PM::powf(&id5, 2.0);

    // let mut loss = id6 * id5;

    // {
    //     // forward pass
    //     println!("Value: id=1 {:?}", id1.value());
    //     println!("Value: id=2 {:?}", id2.value());
    //     println!("Value: id=3 {:?}", id3.value());
    //     println!("Value: id=4 {:?}", id4.value());
    //     println!("Value: id=5 {:?}", id5.value());
    //     println!("Value: id=6 {:?}", id6.value());
    //     println!("Value: loss {:?}", loss.value());

    //     loss.backward();

    //     println!("Grad: id=1 {:?}", id1.grad());
    //     println!("Grad: id=2 {:?}", id2.grad());
    //     println!("Grad: id=3 {:?}", id3.grad());
    //     println!("Grad: id=4 {:?}", id4.grad());
    //     println!("Grad: id=5 {:?}", id5.grad());
    //     println!("Grad: id=6 {:?}", id6.grad());
    //     println!("Grad: loss {:?}", loss.grad());
    // }

    let a = PM::from_value(-4.0);
    let b = PM::from_value(2.0);
    let mut c = a + b;
    let mut d = (a * b) + b.powf(3.0);
    c += c + 1.0;
    c += 1.0 + c + (-a);
    d += d * 2.0 + (b + a).relu();
    d += 3.0 * d + (b - a).relu();

    println!("d={:?}, c={:?}", d, c);
    let e = c - d;
    let f = e.powf(2.0);
    let mut g = f / 2.0;
    g += 10.0 / f;
    assert_eq!(g.value(), 24.704082);

    g.backward();

    println!("a.value = {:?}", a.value());
    println!("b.value = {:?}", b.value());
    println!("c.value = {:?}", c.value());
    println!("d.value = {:?}", d.value());
    println!("e.value = {:?}", e.value());
    println!("f.value = {:?}", f.value());
    println!("g.value = {:?}", g.value());
    println!("=====================================");
    println!("f.grad = {:?}", f.grad());
    println!("e.grad = {:?}", e.grad());
    println!("d.grad = {:?}", d.grad());
    println!("c.grad = {:?}", c.grad());
    println!("a.grad = {:?}", a.grad());
    println!("b.grad = {:?}", b.grad());

    {
        let a = PM::from_value(3.0);

        let b = PM::from_value(2.0);

        let mut c = a + b.exp();

        c.backward();
        // print debug

        println!("a.value = {:?}", a.value());
        println!("b.value = {:?}", b.value());
        println!("c.value = {:?}", c.value());
        println!("=====================================");
        println!("c.grad = {:?}", c.grad());
        println!("b.grad = {:?}", b.grad());
        println!("a.grad = {:?}", a.grad());

        assert_eq!(c.value(), 3.0 + 2.0f32.exp());
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 2.0f32.exp());

        let input_size = 20;
        // generate a random input
        let mut rng = rand::thread_rng();
        let mut input = vec![];
        for _ in 0..input_size {
            input.push(PM::from_value(rng.gen_range(-1.0..1.0)));
        }

        let learning_rate = 0.01;

        let number_of_epochs = 50;
        let mut linear = LinearLayer::new(input_size, 2);
        let mut linear2 = LinearLayer::new(2, input_size);

        for i in 0..number_of_epochs {
            let latent = linear.forward(&input);
            let output = linear2.forward(&latent);
            // calculate the reconstruction loss
            let target = &input;
            let mut loss = PM::from_value(0.0);

            for i in 0..input_size {
                loss += (output[i] - target[i]).powf(2.0);
            }
            loss.backward();

            linear.optimize(learning_rate);
            linear2.optimize(learning_rate);

            linear.zero_grad();
            linear2.zero_grad();
            println!("{}/{} Current loss: {:?}", i+1, number_of_epochs, loss.value());
        }
        // print the latent representation
    }
}

struct LinearLayer {
    input_size: usize,
    output_size: usize,
    params: Vec<param::Param>,
    biases: Vec<param::Param>,
}

impl LinearLayer {
    fn new(input_size: usize, output_size: usize) -> LinearLayer {
        let mut rng = rand::thread_rng();
        let mut params = vec![];
        let mut biases = vec![];
        for _ in 0..output_size {
            for _ in 0..input_size {
                let param = PM::from_value(rng.gen_range(-1.0..1.0));

                params.push(param);
            }
            // params.push(param::PM::add_param(0.0, p, param::Op::Init));
            biases.push(PM::from_value(0.2));
        }
        LinearLayer {
            input_size,
            output_size,
            params,
            biases,
        }
    }

    fn forward(&self, input: &Vec<param::Param>) -> Vec<Param> {
        let mut output = vec![];

        for i in 0..self.output_size {
            let mut sum = param::PM::from_value(0.0);
            for j in 0..self.input_size {
                sum += self.params[i * self.input_size + j] * input[j];
            }
            output.push(sum);
        }

        output
    }

    fn optimize(&mut self, learning_rate: f32) {
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.params[i * self.input_size + j].set_value(
                    self.params[i * self.input_size + j].value()
                        - learning_rate * self.params[i * self.input_size + j].grad(),
                );
            }
        }
    }

    fn zero_grad(&mut self) {
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.params[i * self.input_size + j].set_grad(0.0);
            }
        }
    }
}
