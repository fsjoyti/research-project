/**
 * Copyright (C) 2013-2017 Vasilis Vryniotis <bbriniotis@datumbox.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.datumbox.framework.core.common.interfaces;

/**
 * The Savable interface is implemented by all the objects that can be stored.
 *
 * @author Vasilis Vryniotis <bbriniotis@datumbox.com>
 */
public interface Savable extends AutoCloseable {

    /**
     * Saves the data of the object.
     *
     * @param storageName
     */
    public void save(String storageName);

    /**
     * Deletes the data of the object.
     */
    public void delete();

}
