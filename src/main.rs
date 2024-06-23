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
        Pow,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Param {
        id: i32,
    }

    impl Param {
        // fn id(&self) -> i32 {
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

            // debug info:
            println!("backward_add.id1: {:?}", id1.value());
            println!("backward_add.id2: {:?}", id2.value());
            println!("backward_add.grad: {:?}", grad);

            id1.acc_grad(grad);
            id2.acc_grad(grad);
        }

        pub fn _backward_pow(&self) {
            let prev = self.prev();

            let id1 = prev[0];
            let id2 = prev[1];

            let grad = self.grad();

            // debug prints:

            println!("id1: {:?}", id1.value());
            println!("id2: {:?}", id2.value());
            println!("grad: {:?}", grad);

            println!(
                "acc_grad: {:?}",
                grad * id2.value() * id1.value().powf(id2.value() - 1.0)
            );

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

            println!("_backward_relu.id1: {:?}", id1.value());

            if id1.value() > 0.0 {
                println!("Gradient flowing back: {:?}", self.grad());
                id1.acc_grad(self.grad());
            }
        }

        pub fn _backward(&self) {
            let op = self.op();

            println!("Op: {:?}", op);

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
                _ => {
                    println!("Not implemented");
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
            let mut topo: Vec<Param> = vec![];
            let mut visited: HashSet<i32> = HashSet::new();

            fn build_topo(
                node: &Param,
                topo: &mut Vec<Param>,
                visited: &mut HashSet<i32>,
            ) -> Vec<Param> {

                if !visited.contains(&node.id) {
                    visited.insert(node.id);
                    let prev = node.prev();
                    for prev_node in prev.iter() {
                        println!("Topo before: {:?}", topo);
                        build_topo(prev_node, topo, visited);
                        println!("Topo after: {:?}", topo);
                        
                    }
                    topo.push(node.clone());
                }

                topo.to_vec()
            }
            self.set_grad(1.0);
            let mut topo = build_topo(self, &mut topo, &mut visited);

            println!("Topo: {:?}", topo);
            println!("Topo len: {:?}", topo.len());

            println!("Visited: {:?}", visited);

            let drained = topo.drain(0..topo.len()).collect::<Vec<Param>>();
            let mut reversed = drained;
            reversed.reverse();

            println!("Reversed: {:?}", reversed);

            for node in reversed.drain(..) {
                println!("Before iteration: {:?} {:?}", node, node.grad());
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

        fn next_id(&self) -> i32 {
            self.params.len() as i32
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
    // assert_eq!(c.grad(), 250.0);
}
