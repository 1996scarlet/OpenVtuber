all:
	cd cython/; python3 setup.py build_ext --inplace; rm -rf build; cd ../
	# cd rcnn/pycocotools/; python3 setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd cython/; rm *.so *.c *.cpp; cd ../
	# cd rcnn/pycocotools/; rm *.so; cd ../../
