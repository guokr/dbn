(ns dplearn.base
  (:use  [clj-tuple]
         [clojure.core.matrix]))

(defprotocol Model
  (defn load [this filename])
  (defn save [this filename])
  (defn predicate [this data])
  (defn case-test [this data label])
  (defn fn-predicate [this])
  (defn fn-case-test [this]))

(defprotocol MultiLayers
  (defn pretrain [this k lr data])
  (defn finetune [this lr data label])
  (defn fn-pretrain [this k lr])
  (defn fn-finetune [this lr]))

(defprotocol AutoEncoder
  (defn reconstruct [this]))
  